use assert_cmd::Command;
use predicates::prelude::*;
use profile::collectors::sampling::SAMPLE_COUNT;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::thread;

const MINIMAL_SCRAPE: &str = "# TYPE noop gauge\nnoop 0\n";

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

#[test]
fn diagnose_exits_success() {
    let (url, server) = spawn_metrics_server(MINIMAL_SCRAPE, SAMPLE_COUNT);
    let output = Command::cargo_bin("profile")
        .unwrap()
        .arg("diagnose")
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
        out.contains("PROFILE v") && out.contains('['),
        "stdout should show PROFILE header with model/GPU brackets; got:\n{out}"
    );
    assert!(
        out.contains("GPU =>") && out.contains("UTIL"),
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
        out.contains("PROMPT") && out.contains(" tok"),
        "stdout should include PROMPT row; got:\n{out}"
    );
    assert!(
        out.contains("THROUGHPUT") && out.contains("tok/s"),
        "stdout should include THROUGHPUT row; got:\n{out}"
    );
    assert!(
        out.contains("pfix_cache "),
        "stdout should include pfix_cache % on THROUGHPUT row; got:\n{out}"
    );
    assert!(
        out.contains("Under-batching"),
        "stdout should include rule 1 (Under-batching) section; got:\n{out}"
    );
    assert!(
        out.contains("  - ") || out.contains("ISSUE: Under-batching Detected"),
        "stdout should show rule 1 miss bullets or fired ISSUE block; got:\n{out}"
    );
    assert!(
        out.contains("Not triggered") || out.contains("Under-batching Detected"),
        "stdout should include rule 1 fired or not-triggered title; got:\n{out}"
    );
    assert!(
        out.lines().any(|l| l.starts_with('+') && l.ends_with('+')),
        "stdout should be ASCII-boxed; got:\n{out}"
    );

    server.join().expect("metrics server thread");
}

#[test]
fn diagnose_shows_gen_tok_per_sec_when_counters_increase() {
    const G100: &str = "vllm_generation_tokens_total 100\n";
    const G250: &str = "vllm_generation_tokens_total 250\n";
    let bodies = [G100, G100, G100, G100, G100, G100, G100, G100, G250];
    let (url, server) = spawn_metrics_server_seq(&bodies);
    let output = Command::cargo_bin("profile")
        .unwrap()
        .arg("diagnose")
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
    const G100: &str = "vllm_generation_tokens_total 100\n";
    const G500: &str = "vllm_generation_tokens_total 500\n";
    let bodies = [G500, G500, G500, G500, G500, G500, G500, G500, G100];
    let (url, server) = spawn_metrics_server_seq(&bodies);
    let output = Command::cargo_bin("profile")
        .unwrap()
        .arg("diagnose")
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
        "Usage: profile diagnose [OPTIONS]",
        "-u, --url",
        "vLLM metrics endpoint",
        "[default: http://localhost:8000/metrics]",
        "-m, --max-num-seqs",
        "Engine max_num_seqs if absent on /metrics",
        "[default: 256]",
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
fn verbose_prints_level_to_stderr() {
    let (url, server) = spawn_metrics_server(MINIMAL_SCRAPE, SAMPLE_COUNT);
    Command::cargo_bin("profile")
        .unwrap()
        .args(["-vv", "diagnose", "--url"])
        .arg(&url)
        .assert()
        .success()
        .stderr(predicate::str::contains("Verbose level: 2"));
    server.join().expect("metrics server thread");
}
