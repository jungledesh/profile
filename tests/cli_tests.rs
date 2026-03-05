use assert_cmd::Command;

/// Ensure `profile --help` runs successfully and prints usage.
#[test]
fn help_works() {
    let mut cmd = Command::cargo_bin("profile").unwrap();
    cmd.arg("--help").assert().success();
}

/// Basic smoke test for the `info` subcommand.
#[test]
fn info_subcommand_works() {
    let mut cmd = Command::cargo_bin("profile").unwrap();
    cmd.arg("info").assert().success();
}

