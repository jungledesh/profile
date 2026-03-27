use assert_cmd::Command;
use predicates::prelude::*;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::thread;

/// Minimal exposition text so `prometheus_parse::Scrape::parse` succeeds without vLLM series.
const MINIMAL_SCRAPE: &str = "# TYPE noop gauge\nnoop 0\n";

/// Bind `127.0.0.1:0`, serve one `GET /metrics` response, then finish (for `profile diagnose` tests).
fn spawn_one_shot_metrics_server(body: &'static str) -> (String, thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind test metrics server");
    let port = listener.local_addr().expect("local_addr").port();
    let url = format!("http://127.0.0.1:{port}");

    let handle = thread::spawn(move || {
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
    let (url, server) = spawn_one_shot_metrics_server(MINIMAL_SCRAPE);
    Command::cargo_bin("profile")
        .unwrap()
        .arg("diagnose")
        .arg("--url")
        .arg(&url)
        .assert()
        .success()
        .stdout(predicate::str::contains("PROFILE DIAGNOSE").and(predicate::str::contains("GPU")));
    server.join().expect("metrics server thread");
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
    // -vv must come before the subcommand so it's parsed as a global flag
    let (url, server) = spawn_one_shot_metrics_server(MINIMAL_SCRAPE);
    Command::cargo_bin("profile")
        .unwrap()
        .args(["-vv", "diagnose", "--url"])
        .arg(&url)
        .assert()
        .success()
        .stderr(predicate::str::contains("Verbose level: 2"));
    server.join().expect("metrics server thread");
}
