use assert_cmd::Command;
use predicates::prelude::*;

#[test]
fn help_exits_success() {
    Command::cargo_bin("profile").unwrap()
        .arg("--help")
        .assert()
        .success();
}

#[test]
fn profile_exits_success() {
    Command::cargo_bin("profile").unwrap()
        .arg("profile")
        .assert()
        .success()
        .stdout(predicate::str::contains("(dry-run) would profile vLLM with default configuration"));
}

#[test]
fn profile_with_config_prints_path() {
    Command::cargo_bin("profile").unwrap()
        .args(["profile", "--config", "/path/to/config"])
        .assert()
        .success()
        .stdout(predicate::str::contains("(dry-run) would profile vLLM using config at: /path/to/config"));
}

#[test]
fn info_exits_success() {
    Command::cargo_bin("profile").unwrap()
        .arg("info")
        .assert()
        .success()
        .stdout(predicate::str::contains("profile: Rust CLI for profiling vLLM GPU and system metrics (scaffold)"));
}

#[test]
fn verbose_prints_level_to_stderr() {
    // -vv must come before the subcommand so it's parsed as a global flag
    Command::cargo_bin("profile").unwrap()
        .args(["-vv", "profile"])
        .assert()
        .success()
        .stderr(predicate::str::contains("Verbose level: 2"));
}
