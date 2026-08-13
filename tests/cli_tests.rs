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
fn root_help_lists_commands_and_options() {
    let output = Command::cargo_bin("profile")
        .unwrap()
        .arg("--help")
        .output()
        .expect("run profile --help");
    assert!(
        output.status.success(),
        "stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let out = String::from_utf8_lossy(&output.stdout).into_owned();
    let version_line = format!("profile {}", env!("CARGO_PKG_VERSION"));
    for needle in [
        version_line.as_str(),
        "Detects inefficiencies. Suggests fixes.",
        "Usage: profile <COMMAND> [OPTIONS]",
        "diagnose",
        "help",
        "completions",
        "man",
        "-u, --url",
        "[env: PROFILE_URL]",
        "--duration",
        "[env: PROFILE_DURATION]",
        "-m, --max-num-seqs",
        "[env: PROFILE_MAX_NUM_SEQS]",
        "--tensor-parallel-size",
        "[env: PROFILE_TENSOR_PARALLEL_SIZE]",
        "--cost-per-hour",
        "[env: PROFILE_COST_PER_HOUR]",
        "-v, --verbose",
        "[env: PROFILE_VERBOSE]",
        "-h, --help",
    ] {
        assert!(
            out.contains(needle),
            "profile --help should mention {needle:?}; got:\n{out}"
        );
    }
}

#[test]
fn version_flag_prints_crate_version() {
    let expected = format!("profile {}", env!("CARGO_PKG_VERSION"));
    for args in [&["--version"][..], &["-V"][..]] {
        let output = Command::cargo_bin("profile")
            .unwrap()
            .args(args)
            .output()
            .expect("run profile --version");
        assert!(
            output.status.success(),
            "args {args:?} stderr:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let out = String::from_utf8_lossy(&output.stdout);
        assert!(
            out.contains(&expected),
            "args {args:?} should print {expected:?}; got:\n{out}"
        );
    }
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
    let version_line = format!("profile {}", env!("CARGO_PKG_VERSION"));
    for needle in [
        version_line.as_str(),
        "Collects metrics. Detects inefficiencies. Suggests fixes.",
        "A loop: collect, you apply a fix, remeasure. Not a one-shot dump.",
        "Pass -v",
        "Usage: profile diagnose [OPTIONS]",
        "-u, --url",
        "[env: PROFILE_URL]",
        "vLLM metrics endpoint",
        "[default: http://localhost:8000/metrics]",
        "-m, --max-num-seqs",
        "Engine max_num_seqs (auto-detected from /metrics when available; prompted if absent)",
        "--duration",
        "m  minutes (not ms, not \"mins\", not bare m)",
        "2m    short run",
        "10m   longer run",
        "30m   maximum",
        "[default: 30s]",
        "-v, --verbose",
        "Show rules that did not fire, physics limits, and extra GPU, latency, cache, and config detail",
        "--tensor-parallel-size",
        "--cost-per-hour",
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
fn completions_bash_prints_script() {
    let output = Command::cargo_bin("profile")
        .unwrap()
        .args(["completions", "bash"])
        .output()
        .expect("run profile completions bash");
    assert!(
        output.status.success(),
        "stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let out = String::from_utf8_lossy(&output.stdout);
    assert!(
        out.contains("profile"),
        "bash completion should mention profile; got:\n{out}"
    );
}

#[test]
fn man_prints_groff() {
    let output = Command::cargo_bin("profile")
        .unwrap()
        .arg("man")
        .output()
        .expect("run profile man");
    assert!(
        output.status.success(),
        "stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let out = String::from_utf8_lossy(&output.stdout);
    assert!(
        out.contains(".TH profile"),
        "man page should start with a groff title; got:\n{out}"
    );
}
