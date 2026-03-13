use assert_cmd::Command;
use predicates::prelude::*;

#[test]
fn help_exits_success() {
    Command::cargo_bin("profile")
        .unwrap()
        .arg("--help")
        .assert()
        .success();
}

#[test]
fn run_exits_success() {
    Command::cargo_bin("profile")
        .unwrap()
        .arg("run")
        .assert()
        .success()
        .stdout(predicate::str::contains("Profile v1").and(predicate::str::contains("GPU")));
}

#[test]
fn run_with_config_succeeds() {
    Command::cargo_bin("profile")
        .unwrap()
        .args(["run", "--config", "/path/to/config"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Profile v1"));
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
    Command::cargo_bin("profile")
        .unwrap()
        .args(["-vv", "run"])
        .assert()
        .success()
        .stderr(predicate::str::contains("Verbose level: 2"));
}
