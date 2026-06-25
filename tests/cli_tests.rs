use assert_cmd::Command;

#[test]
fn help_exits_success() {
    Command::cargo_bin("profile")
        .unwrap()
        .arg("--help")
        .assert()
        .success();
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
        "Engine max_num_seqs (auto-detected from /metrics when available; prompted if absent)",
        "--duration",
        "m  minutes (not ms, not \"mins\", not bare m)",
        "1m    one-minute run",
        "2m    short run",
        "3m    maximum",
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
