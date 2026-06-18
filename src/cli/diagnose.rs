use std::io::{self, BufRead, Write};
use std::time::Duration;

use crate::{context, engine, output, profiler};

/// Fast-fail before collection starts. Longer hang is user-visible at startup.
const PRE_FLIGHT_TIMEOUT: Duration = Duration::from_secs(2);

pub fn execute(
    vllm_metrics_input: &str,
    max_num_seqs: Option<u32>,
    cost_per_hour: Option<f64>,
    tensor_parallel_size: Option<u32>,
    verbose_rules: bool,
    duration: Duration,
) -> anyhow::Result<()> {
    let resolved: Option<u32> = if max_num_seqs.is_some() {
        max_num_seqs
    } else {
        match pre_flight_max_num_seqs(vllm_metrics_input) {
            Some(v) => Some(v),
            None => Some(prompt_for_max_num_seqs()?),
        }
    };

    let result = profiler::run_diagnose(
        vllm_metrics_input,
        resolved,
        cost_per_hour,
        tensor_parallel_size,
        duration,
    )?;
    output::stdout::print_diagnose_table(&result, verbose_rules);

    if !result.any_evaluable {
        return Ok(());
    }

    let aggregate_win = context::RuntimeWindow::from_snapshot(result.snapshot.clone());
    let summary_input = context::AnalysisInput::new(&result.static_ctx, &aggregate_win);
    let report = engine::build_report_for_diagnose(&result.windows, summary_input);

    profiler::loop_runner::run(
        vllm_metrics_input,
        resolved,
        cost_per_hour,
        tensor_parallel_size,
        duration,
        result,
        report,
    )?;

    Ok(())
}

fn pre_flight_max_num_seqs(url: &str) -> Option<u32> {
    let client = reqwest::blocking::Client::builder()
        .use_rustls_tls()
        .timeout(PRE_FLIGHT_TIMEOUT)
        .build()
        .ok()?;
    let metrics_url = crate::collectors::vllm::metrics_url(url);
    let body = crate::collectors::vllm::fetch_metrics_body(&client, &metrics_url).ok()?;
    let scrape = crate::collectors::vllm::scrape_from_body(&body).ok()?;
    crate::collectors::vllm::max_num_seqs_from_scrape(&scrape)
}

const MAX_NUM_SEQS_PROMPT: &str =
    "--max-num-seqs [Hint: check your vLLM start command] (default 256): ";

fn prompt_for_max_num_seqs() -> anyhow::Result<u32> {
    let v = prompt_u32_with_default(
        &mut io::stdin().lock(),
        &mut io::stdout(),
        256,
        MAX_NUM_SEQS_PROMPT,
    )?;
    println!();
    Ok(v)
}

pub(crate) fn prompt_for_updated_max_num_seqs(
    current: u32,
    stdin_rx: &std::sync::mpsc::Receiver<String>,
) -> anyhow::Result<u32> {
    let prompt = updated_max_num_seqs_prompt(current);
    prompt_u32_from_channel(stdin_rx, &mut io::stdout(), current, &prompt)
}

fn updated_max_num_seqs_prompt(current: u32) -> String {
    format!("New --max-num-seqs [current: {current}]: ")
}

fn retry_u32_loop<F, W>(
    writer: &mut W,
    default: u32,
    prompt: &str,
    mut next_line: F,
) -> anyhow::Result<u32>
where
    F: FnMut() -> anyhow::Result<String>,
    W: Write,
{
    const MAX_ATTEMPTS: u8 = 4;
    let mut attempts: u8 = 0;
    loop {
        write!(writer, "{prompt}")?;
        writer.flush()?;
        let line = next_line()?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            return Ok(default);
        }
        match trimmed.parse::<u32>() {
            Ok(0) => {
                attempts += 1;
                if attempts >= MAX_ATTEMPTS {
                    return Err(anyhow::anyhow!(
                        "max_num_seqs: too many invalid attempts. Pass -m <value> to skip the prompt."
                    ));
                }
                writeln!(writer, "max_num_seqs must be greater than zero.")?;
            }
            Ok(v) => return Ok(v),
            Err(_) => {
                attempts += 1;
                if attempts >= MAX_ATTEMPTS {
                    return Err(anyhow::anyhow!(
                        "max_num_seqs: too many invalid attempts. Pass -m <value> to skip the prompt."
                    ));
                }
                writeln!(writer, "expected a positive integer, got {trimmed:?}")?;
            }
        }
    }
}

fn prompt_u32_with_default<R: BufRead, W: Write>(
    reader: &mut R,
    writer: &mut W,
    default: u32,
    prompt: &str,
) -> anyhow::Result<u32> {
    retry_u32_loop(writer, default, prompt, || {
        let mut l = String::new();
        reader.read_line(&mut l)?;
        Ok(l)
    })
}

fn prompt_u32_from_channel<W: Write>(
    stdin_rx: &std::sync::mpsc::Receiver<String>,
    writer: &mut W,
    default: u32,
    prompt: &str,
) -> anyhow::Result<u32> {
    retry_u32_loop(writer, default, prompt, || {
        stdin_rx.recv().map_err(|_| anyhow::anyhow!("stdin closed"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn run(input: &[u8], default: u32) -> anyhow::Result<u32> {
        let mut out = Vec::new();
        prompt_u32_with_default(
            &mut Cursor::new(input),
            &mut out,
            default,
            MAX_NUM_SEQS_PROMPT,
        )
    }

    #[test]
    fn empty_input_returns_default() {
        assert_eq!(run(b"\n", 256).unwrap(), 256);
    }

    #[test]
    fn valid_input_returns_value() {
        assert_eq!(run(b"128\n", 256).unwrap(), 128);
    }

    #[test]
    fn three_bad_inputs_does_not_exit() {
        // 3 bad + 1 valid — should succeed
        assert_eq!(run(b"0\n0\n0\n64\n", 256).unwrap(), 64);
    }

    #[test]
    fn four_bad_inputs_exits_with_error() {
        assert!(run(b"0\n0\n0\n0\n", 256).is_err());
    }

    #[test]
    fn four_non_numeric_inputs_exits_with_error() {
        assert!(run(b"abc\nabc\nabc\nabc\n", 256).is_err());
    }

    #[test]
    fn valid_input_after_one_bad_attempt_succeeds() {
        assert_eq!(run(b"0\n64\n", 256).unwrap(), 64);
    }

    #[test]
    fn error_message_contains_recovery_hint() {
        let err = run(b"0\n0\n0\n0\n", 256).unwrap_err();
        assert!(err.to_string().contains("Pass -m <value>"));
    }

    #[test]
    fn updated_max_num_seqs_prompt_labels_current_value() {
        assert_eq!(
            updated_max_num_seqs_prompt(128),
            "New --max-num-seqs [current: 128]: "
        );
    }

    #[test]
    fn updated_max_num_seqs_empty_input_keeps_current() {
        let (tx, rx) = std::sync::mpsc::channel();
        tx.send(String::new()).unwrap();
        let mut out = Vec::new();
        let current = 512;
        let prompt = updated_max_num_seqs_prompt(current);
        let v = prompt_u32_from_channel(&rx, &mut out, current, &prompt).unwrap();
        assert_eq!(v, 512);
        let written = String::from_utf8(out).unwrap();
        assert!(written.contains("New --max-num-seqs [current: 512]: "));
    }

    #[test]
    fn updated_max_num_seqs_valid_input_returns_new_value() {
        let (tx, rx) = std::sync::mpsc::channel();
        tx.send("64".to_string()).unwrap();
        let mut out = Vec::new();
        let prompt = updated_max_num_seqs_prompt(256);
        let v = prompt_u32_from_channel(&rx, &mut out, 256, &prompt).unwrap();
        assert_eq!(v, 64);
    }
}
