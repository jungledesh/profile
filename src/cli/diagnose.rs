use std::io::{self, BufRead, Write};
use std::time::Duration;

use crate::cli::gpu_assignment::resolve_gpu_assignment;
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
    let resolved: u32 = if let Some(v) = max_num_seqs {
        v
    } else {
        match pre_flight_max_num_seqs(vllm_metrics_input) {
            Some(v) => v,
            None => prompt_for_max_num_seqs()?,
        }
    };

    let assignment = resolve_gpu_assignment(tensor_parallel_size, vllm_metrics_input)?;

    let result = profiler::run_diagnose(
        vllm_metrics_input,
        Some(resolved),
        cost_per_hour,
        assignment.tp,
        assignment.indices.clone(),
        duration,
    )?;

    let aggregate_win = context::RuntimeWindow::from_snapshot(result.snapshot.clone());
    let summary_input = context::AnalysisInput::new(&result.static_ctx, &aggregate_win);
    let report = engine::build_report_for_diagnose(&result.windows, summary_input);
    output::stdout::print_diagnose_table_with_report(
        &result,
        &report,
        &aggregate_win,
        verbose_rules,
        false,
    );

    // Incomplete measurement: table already printed. Do not start the closed loop
    // (avoids empty recommendations → false healthy exit).
    if !result.any_evaluable
        || result.all_idle
        || report.n_eval < engine::ENGINE_MIN_PERSISTENT_WINDOWS
    {
        return Ok(());
    }

    profiler::loop_runner::run(profiler::loop_runner::LoopRunnerInput {
        url: vllm_metrics_input,
        max_num_seqs: resolved,
        cost_per_hour,
        tensor_parallel_size: assignment.tp,
        gpu_indices: assignment.indices,
        duration,
        initial_result: result,
        initial_report: report,
        verbose_rules,
        max_num_seqs_prompt: &mut DiagnoseMaxNumSeqsPrompt,
    })?;

    Ok(())
}

struct DiagnoseMaxNumSeqsPrompt;

impl profiler::MaxNumSeqsPrompt for DiagnoseMaxNumSeqsPrompt {
    fn ask(
        &mut self,
        current: u32,
        stdin_rx: &std::sync::mpsc::Receiver<String>,
    ) -> anyhow::Result<u32> {
        prompt_for_updated_max_num_seqs(current, stdin_rx)
    }
}

fn pre_flight_max_num_seqs(url: &str) -> Option<u32> {
    crate::collectors::vllm::preflight_max_num_seqs(url, PRE_FLIGHT_TIMEOUT)
}

const MAX_NUM_SEQS_PROMPT: &str =
    "--max-num-seqs [Hint: check your vLLM start command] (default 256): ";

pub(crate) const TP_ABORT_HINT: &str = "Pass --tensor-parallel-size <value> to skip the prompt.";

fn prompt_for_max_num_seqs() -> anyhow::Result<u32> {
    println!();
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
    default: Option<u32>,
    abort_hint: &str,
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
            if let Some(v) = default {
                return Ok(v);
            }
            attempts += 1;
            if attempts >= MAX_ATTEMPTS {
                return Err(anyhow::anyhow!("Too many invalid attempts. {abort_hint}"));
            }
            writeln!(writer, "enter a positive integer.")?;
            continue;
        }
        match trimmed.parse::<u32>() {
            Ok(0) => {
                attempts += 1;
                if attempts >= MAX_ATTEMPTS {
                    return Err(anyhow::anyhow!("Too many invalid attempts. {abort_hint}"));
                }
                writeln!(writer, "value must be greater than zero.")?;
            }
            Ok(v) => return Ok(v),
            Err(_) => {
                attempts += 1;
                if attempts >= MAX_ATTEMPTS {
                    return Err(anyhow::anyhow!("Too many invalid attempts. {abort_hint}"));
                }
                writeln!(writer, "expected a positive integer, got {trimmed:?}")?;
            }
        }
    }
}

const MAX_NUM_SEQS_ABORT_HINT: &str = "Pass -m <value> to skip the prompt.";

fn prompt_u32_with_default<R: BufRead, W: Write>(
    reader: &mut R,
    writer: &mut W,
    default: u32,
    prompt: &str,
) -> anyhow::Result<u32> {
    retry_u32_loop(
        writer,
        Some(default),
        MAX_NUM_SEQS_ABORT_HINT,
        prompt,
        || {
            let mut l = String::new();
            reader.read_line(&mut l)?;
            Ok(l)
        },
    )
}

pub(crate) fn prompt_u32_required<R: BufRead, W: Write>(
    reader: &mut R,
    writer: &mut W,
    prompt: &str,
) -> anyhow::Result<u32> {
    retry_u32_loop(writer, None, TP_ABORT_HINT, prompt, || {
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
    retry_u32_loop(
        writer,
        Some(default),
        MAX_NUM_SEQS_ABORT_HINT,
        prompt,
        || stdin_rx.recv().map_err(|_| anyhow::anyhow!("stdin closed")),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn run_max_num_seqs(input: &[u8], default: u32) -> anyhow::Result<u32> {
        let mut out = Vec::new();
        prompt_u32_with_default(
            &mut Cursor::new(input),
            &mut out,
            default,
            MAX_NUM_SEQS_PROMPT,
        )
    }

    fn run_tp(input: &[u8], host_gpus: u32) -> anyhow::Result<u32> {
        let mut out = Vec::new();
        prompt_u32_required(
            &mut Cursor::new(input),
            &mut out,
            &format!("Enter value for `--tensor-parallel-size ({host_gpus} gpus detected)` : "),
        )
    }

    #[test]
    fn empty_input_returns_default() {
        assert_eq!(run_max_num_seqs(b"\n", 256).unwrap(), 256);
    }

    #[test]
    fn valid_input_returns_value() {
        assert_eq!(run_max_num_seqs(b"128\n", 256).unwrap(), 128);
    }

    #[test]
    fn three_bad_inputs_does_not_exit() {
        assert_eq!(run_max_num_seqs(b"0\n0\n0\n64\n", 256).unwrap(), 64);
    }

    #[test]
    fn four_bad_inputs_exits_with_error() {
        assert!(run_max_num_seqs(b"0\n0\n0\n0\n", 256).is_err());
    }

    #[test]
    fn four_non_numeric_inputs_exits_with_error() {
        assert!(run_max_num_seqs(b"abc\nabc\nabc\nabc\n", 256).is_err());
    }

    #[test]
    fn valid_input_after_one_bad_attempt_succeeds() {
        assert_eq!(run_max_num_seqs(b"0\n64\n", 256).unwrap(), 64);
    }

    #[test]
    fn error_message_contains_recovery_hint() {
        let err = run_max_num_seqs(b"0\n0\n0\n0\n", 256).unwrap_err();
        assert!(err.to_string().contains("Pass -m <value>"));
    }

    #[test]
    fn tp_empty_input_reprompts_until_valid() {
        assert_eq!(run_tp(b"\n\n2\n", 8).unwrap(), 2);
    }

    #[test]
    fn tp_empty_input_aborts_after_four_attempts() {
        let err = run_tp(b"\n\n\n\n", 8).unwrap_err();
        assert!(err.to_string().contains("Pass --tensor-parallel-size"));
    }

    #[test]
    fn tp_valid_input_returns_value() {
        assert_eq!(run_tp(b"2\n", 8).unwrap(), 2);
    }

    #[test]
    fn tp_zero_reprompts() {
        assert_eq!(run_tp(b"0\n4\n", 8).unwrap(), 4);
    }

    #[test]
    fn tp_non_numeric_reprompts() {
        assert_eq!(run_tp(b"abc\n2\n", 8).unwrap(), 2);
    }

    #[test]
    fn tp_four_bad_inputs_aborts_with_hint() {
        let err = run_tp(b"0\n0\n0\n0\n", 8).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("Too many invalid attempts"));
        assert!(msg.contains("Pass --tensor-parallel-size"));
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
