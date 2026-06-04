use assert_cmd::Command;
use predicates::prelude::*;
use profile::collectors::sampling::SAMPLE_COUNT;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::thread;

const MINIMAL_SCRAPE: &str = "# TYPE noop gauge\nnoop 0\n";

/// Includes `vllm_num_requests_running` so the window passes `window_is_evaluable`.
const MINIMAL_EVALUABLE_SCRAPE: &str = r#"# TYPE vllm_num_requests_running gauge
vllm_num_requests_running 20
vllm_max_num_seqs 256
vllm_kv_cache_usage_perc 10
vllm_prefix_cache_hits_total 50
vllm_prefix_cache_queries_total 100
vllm_request_success_total 10
vllm_generation_tokens_total 1000
# TYPE noop gauge
noop 0
"#;

fn spawn_metrics_server(
    body: &'static str,
    response_count: usize,
) -> (String, thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind test metrics server");
    let port = listener.local_addr().expect("local_addr").port();
    let url = format!("http://127.0.0.1:{port}");

    let handle = thread::spawn(move || {
        for _ in 0..response_count {
            let (mut stream, _) = listener.accept().expect("accept");
            let mut buf = [0u8; 4096];
            let mut n = 0usize;
            while n < buf.len() {
                let got = stream.read(&mut buf[n..]).expect("read");
                if got == 0 {
                    break;
                }
                n += got;
                if buf[..n].windows(4).any(|w| w == b"\r\n\r\n") {
                    break;
                }
            }
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            stream.write_all(resp.as_bytes()).expect("write response");
        }
    });

    (url, handle)
}

/// One response body per GET; length must match [`SAMPLE_COUNT`].
fn spawn_metrics_server_seq(bodies: &[&'static str]) -> (String, thread::JoinHandle<()>) {
    assert_eq!(
        bodies.len(),
        SAMPLE_COUNT,
        "vLLM collector performs exactly {SAMPLE_COUNT} scrapes"
    );
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind test metrics server");
    let port = listener.local_addr().expect("local_addr").port();
    let url = format!("http://127.0.0.1:{port}");
    let bodies: Vec<&'static str> = bodies.to_vec();

    let handle = thread::spawn(move || {
        for body in bodies {
            let (mut stream, _) = listener.accept().expect("accept");
            let mut buf = [0u8; 4096];
            let mut n = 0usize;
            while n < buf.len() {
                let got = stream.read(&mut buf[n..]).expect("read");
                if got == 0 {
                    break;
                }
                n += got;
                if buf[..n].windows(4).any(|w| w == b"\r\n\r\n") {
                    break;
                }
            }
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            stream.write_all(resp.as_bytes()).expect("write response");
        }
    });

    (url, handle)
}

#[test]
fn help_exits_success() {
    Command::cargo_bin("profile")
        .unwrap()
        .arg("--help")
        .assert()
        .success();
}

const MINIMAL_PREFIX_EARLY: &str = r#"vllm_prefix_cache_hits 0
vllm_prefix_cache_queries 100
vllm_num_requests_running 20
vllm_max_num_seqs 256
vllm_kv_cache_usage_perc 50
vllm_request_success_total 10
vllm_generation_tokens_total 1000
"#;
const MINIMAL_PREFIX_LATE: &str = r#"vllm_prefix_cache_hits 50
vllm_prefix_cache_queries 200
vllm_num_requests_running 20
vllm_max_num_seqs 256
vllm_kv_cache_usage_perc 50
vllm_request_success_total 10
vllm_generation_tokens_total 1100
"#;

#[test]
fn diagnose_exits_success() {
    let bodies = [
        MINIMAL_PREFIX_EARLY,
        MINIMAL_PREFIX_EARLY,
        MINIMAL_PREFIX_EARLY,
        MINIMAL_PREFIX_EARLY,
        MINIMAL_PREFIX_EARLY,
        MINIMAL_PREFIX_EARLY,
        MINIMAL_PREFIX_EARLY,
        MINIMAL_PREFIX_EARLY,
        MINIMAL_PREFIX_LATE,
    ];
    let (url, server) = spawn_metrics_server_seq(&bodies);
    let output = Command::cargo_bin("profile")
        .unwrap()
        .args(["diagnose", "--duration", "2s", "-m", "256", "--url"])
        .arg(&url)
        .output()
        .expect("run profile diagnose");

    assert!(
        output.status.success(),
        "stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let out = String::from_utf8_lossy(&output.stdout).into_owned();
    assert!(
        !out.contains("[i]"),
        "minimal scrape should not emit metric advisories; got:\n{out}"
    );
    assert!(
        out.contains("PROFILE v") && out.contains('[') && out.contains(" UTC]"),
        "stdout should show PROFILE header with model/GPU and bracketed UTC timestamp; got:\n{out}"
    );
    assert!(
        out.contains("GPU =>") && out.contains("EFFICIENCY"),
        "stdout should include GPU => row; got:\n{out}"
    );
    assert!(
        out.contains("vLLM:"),
        "stdout should include vLLM: header; got:\n{out}"
    );
    assert!(
        out.contains("REQUESTS") && out.contains("run "),
        "stdout should include REQUESTS row; got:\n{out}"
    );
    assert!(
        out.contains("LATENCY") && out.contains("ttft "),
        "stdout should include LATENCY row; got:\n{out}"
    );
    assert!(
        out.contains("PROMPT") && out.contains("kv_cache"),
        "stdout should include PROMPT row; got:\n{out}"
    );
    assert!(
        out.contains("THROUGHPUT") && out.contains("tok/s"),
        "stdout should include THROUGHPUT row; got:\n{out}"
    );
    assert!(
        out.contains("TRAFFIC"),
        "stdout should always include TRAFFIC row; got:\n{out}"
    );
    assert!(
        out.contains("pfix_cache "),
        "stdout should include pfix_cache % on THROUGHPUT row; got:\n{out}"
    );
    assert!(
        out.contains("No issues detected in this snapshot."),
        "default diagnose should report no issues when nothing fires; got:\n{out}"
    );
    assert!(
        !out.contains("ISSUE:") && !out.contains("not indicated"),
        "default diagnose should omit ISSUE and verbose-only lines; got:\n{out}"
    );
    assert!(
        out.lines().any(|l| l.starts_with('+') && l.ends_with('+')),
        "stdout should be ASCII-boxed; got:\n{out}"
    );

    server.join().expect("metrics server thread");
}

#[test]
fn diagnose_shows_gen_tok_per_sec_when_counters_increase() {
    const G100: &str = "vllm_num_requests_running 0\nvllm_generation_tokens_total 100\n";
    const G250: &str = "vllm_num_requests_running 0\nvllm_generation_tokens_total 250\n";
    let bodies = [G100, G100, G100, G100, G100, G100, G100, G100, G250];
    let (url, server) = spawn_metrics_server_seq(&bodies);
    let output = Command::cargo_bin("profile")
        .unwrap()
        .arg("diagnose")
        .arg("--duration")
        .arg("2s")
        .arg("--url")
        .arg(&url)
        .output()
        .expect("run profile diagnose");

    assert!(
        output.status.success(),
        "stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let out = String::from_utf8_lossy(&output.stdout).into_owned();
    assert!(
        out.lines().any(|line| {
            line.contains("THROUGHPUT") && line.contains("tok/s") && !line.contains("— tok/s")
        }),
        "expected THROUGHPUT row with numeric tok/s; got:\n{out}"
    );
    server.join().expect("metrics server thread");
}

#[test]
fn diagnose_gen_tok_per_sec_na_when_counter_resets() {
    // Keep window evaluable via running count; counter reset still yields NA tok/s (invalid Δ).
    const B500: &str = concat!(
        "vllm_num_requests_running 20\n",
        "vllm_generation_tokens_total 500\n"
    );
    const B100: &str = concat!(
        "vllm_num_requests_running 20\n",
        "vllm_generation_tokens_total 100\n"
    );
    let bodies = [B500, B500, B500, B500, B500, B500, B500, B500, B100];
    let (url, server) = spawn_metrics_server_seq(&bodies);
    let output = Command::cargo_bin("profile")
        .unwrap()
        .arg("diagnose")
        .arg("--duration")
        .arg("2s")
        .arg("--url")
        .arg(&url)
        .output()
        .expect("run profile diagnose");

    assert!(
        output.status.success(),
        "stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let out = String::from_utf8_lossy(&output.stdout).into_owned();
    assert!(
        out.lines()
            .any(|line| line.contains("THROUGHPUT") && line.contains("— tok/s")),
        "expected THROUGHPUT row with — tok/s after invalid delta; got:\n{out}"
    );
    server.join().expect("metrics server thread");
}

#[test]
fn diagnose_verbose_shows_not_indicated_lines() {
    let (url, server) = spawn_metrics_server(MINIMAL_EVALUABLE_SCRAPE, SAMPLE_COUNT);
    let output = Command::cargo_bin("profile")
        .unwrap()
        .args(["-v", "diagnose", "--duration", "2s", "--url"])
        .arg(&url)
        .output()
        .expect("run profile -v diagnose");

    assert!(
        output.status.success(),
        "stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let out = String::from_utf8_lossy(&output.stdout).into_owned();
    assert!(
        out.contains("Under-batching: not triggered")
            && out.contains("KV cache pressure: not triggered")
            && out.contains("Prefix cache hit rate: not triggered")
            && out.contains("Concurrency saturation: not triggered")
            && !out.contains("No issues detected in this snapshot."),
        "expected verbose rule status lines without redundant no-issues summary; got:\n{out}"
    );
    server.join().expect("metrics server thread");
}

#[test]
fn diagnose_help_lists_usage_and_options() {
    let output = Command::cargo_bin("profile")
        .unwrap()
        .args(["diagnose", "--help"])
        .output()
        .expect("run profile diagnose --help");

    assert!(
        output.status.success(),
        "stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let out = String::from_utf8_lossy(&output.stdout).into_owned();
    for needle in [
        "Collects metrics. Detects inefficiencies. Suggests fixes.",
        "Pass -v",
        "Usage: profile diagnose [OPTIONS]",
        "-u, --url",
        "vLLM metrics endpoint",
        "[default: http://localhost:8000/metrics]",
        "-m, --max-num-seqs",
        "Engine max_num_seqs if absent on /metrics",
        "[default: 256]",
        "--duration",
        "m  minutes",
        "not ms (milliseconds)",
        "2m    short run",
        "10m   sustained observation",
        "30m   long soak",
        "[default: 30s]",
        "-h, --help",
        "Display this message",
    ] {
        assert!(
            out.contains(needle),
            "diagnose --help should mention {needle:?}; got:\n{out}"
        );
    }
}

#[test]
fn r5_concurrency_saturation_fires_in_output() {
    // run >= max_num_seqs (32) with >30% queue ratio → r5 should fire
    const SCRAPE: &str = r#"# TYPE vllm_num_requests_running gauge
vllm_num_requests_running 32
# TYPE vllm_num_requests_waiting gauge
vllm_num_requests_waiting 15
# TYPE vllm_max_num_seqs gauge
vllm_max_num_seqs 32
# TYPE vllm_generation_tokens_total counter
vllm_generation_tokens_total 1000
"#;
    let bodies = [SCRAPE; SAMPLE_COUNT];
    let (url, server) = spawn_metrics_server_seq(&bodies);
    let output = Command::cargo_bin("profile")
        .unwrap()
        .args(["diagnose", "--duration", "2s", "--url"])
        .arg(&url)
        .output()
        .expect("run profile diagnose");

    let out = String::from_utf8_lossy(&output.stdout).into_owned();
    assert!(
        out.contains("Concurrency Saturation"),
        "expected r5 Concurrency Saturation in output; got:\n{out}"
    );
    assert!(
        out.contains("--max-num-seqs=32"),
        "output should show max-num-seqs cap; got:\n{out}"
    );
    server.join().expect("metrics server thread");
}

#[test]
fn verbose_prints_level_to_stderr() {
    let (url, server) = spawn_metrics_server(MINIMAL_SCRAPE, SAMPLE_COUNT);
    Command::cargo_bin("profile")
        .unwrap()
        .args(["-vv", "diagnose", "--duration", "2s", "--url"])
        .arg(&url)
        .assert()
        .success()
        .stderr(predicate::str::contains("Verbose level: 2"));
    server.join().expect("metrics server thread");
}
