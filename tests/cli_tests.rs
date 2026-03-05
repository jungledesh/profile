use assert_cmd::Command;
use predicates::prelude::*;

#[test]
fn help_exits_success() {
    Command::cargo_bin("infer-pro").unwrap()
        .arg("--help")
        .assert()
        .success();
}

#[test]
fn profile_exits_success() {
    Command::cargo_bin("infer-pro").unwrap()
        .arg("profile")
        .assert()
        .success();
}

#[test]
fn profile_with_config_prints_path() {
    Command::cargo_bin("infer-pro").unwrap()
        .args(["profile", "--config", "/path/to/config"])
        .assert()
        .success()
        .stdout(predicate::str::contains("/path/to/config"));
}

#[test]
fn info_exits_success() {
    Command::cargo_bin("infer-pro").unwrap()
        .arg("info")
        .assert()
        .success();
}
