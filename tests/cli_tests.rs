use assert_cmd::Command;
use predicates::prelude::*;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::thread;

const MINIMAL_SCRAPE: &str = "# TYPE noop gauge\nnoop 0\n";

/// vLLM collector issues `count` sequential GETs to `/metrics` (default 8).
const VLLM_SCRAPE_COUNT: usize = 8;

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

/// One response body per GET; must match [`VLLM_SCRAPE_COUNT`].
fn spawn_metrics_server_seq(bodies: &[&'static str]) -> (String, thread::JoinHandle<()>) {
    assert_eq!(
        bodies.len(),
        VLLM_SCRAPE_COUNT,
        "vLLM collector performs exactly {VLLM_SCRAPE_COUNT} scrapes"
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
    let (url, server) = spawn_metrics_server(MINIMAL_SCRAPE, VLLM_SCRAPE_COUNT);
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
        out.contains("GPU name"),
        "stdout should list GPU name; got:\n{out}"
    );
    assert!(
        out.lines()
            .any(|line| line.contains("GPU name") && line.contains(" : ")),
        "expected aligned `label : value` line for GPU name; got:\n{out}"
    );

    server.join().expect("metrics server thread");
}

#[test]
fn diagnose_shows_gen_tok_per_sec_when_counters_increase() {
    const G100: &str = "vllm_generation_tokens_total 100\n";
    const G250: &str = "vllm_generation_tokens_total 250\n";
    let bodies = [G100, G100, G100, G100, G100, G100, G100, G250];
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
            .any(|line| line.contains("Gen tok/s") && !line.contains("(n/a)")),
        "expected Gen tok/s with a numeric rate; got:\n{out}"
    );
    server.join().expect("metrics server thread");
}

#[test]
fn diagnose_gen_tok_per_sec_na_when_counter_resets() {
    const G100: &str = "vllm_generation_tokens_total 100\n";
    const G500: &str = "vllm_generation_tokens_total 500\n";
    let bodies = [G500, G500, G500, G500, G500, G500, G500, G100];
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
            .any(|line| line.contains("Gen tok/s") && line.contains("(n/a)")),
        "expected Gen tok/s (n/a) after negative delta; got:\n{out}"
    );
    server.join().expect("metrics server thread");
}

#[test]
fn diagnose_long_help_lists_example_metrics() {
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
        "Example:",
        "GPU util %",
        "Mem ctrl util %",
        "VRAM % used",
        "Power draw",
        "SM clock",
        "In-batch reqs",
        "Waiting reqs",
        "Max seqs",
        "TTFT (est. ms)",
        "Prefill ms",
        "Queue ms",
        "TPOT ms",
        "Prompt mean",
        "Gen tok/s",
        "Prefix cache hit rate (last scrape)",
        "Gen tokens",
    ] {
        assert!(
            out.contains(needle),
            "diagnose --help should mention {needle:?}; got:\n{out}"
        );
    }
}

#[test]
fn info_exits_success() {
    Command::cargo_bin("profile")
        .unwrap()
        .arg("info")
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "profile: Rust CLI for profiling vLLM GPU and system metrics (scaffold)",
        ));
}

#[test]
fn verbose_prints_level_to_stderr() {
    let (url, server) = spawn_metrics_server(MINIMAL_SCRAPE, VLLM_SCRAPE_COUNT);
    Command::cargo_bin("profile")
        .unwrap()
        .args(["-vv", "diagnose", "--url"])
        .arg(&url)
        .assert()
        .success()
        .stderr(predicate::str::contains("Verbose level: 2"));
    server.join().expect("metrics server thread");
}
