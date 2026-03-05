use assert_cmd::Command;

/// Ensure `infer-pro --help` runs successfully and prints usage.
#[test]
fn help_works() {
    let mut cmd = Command::cargo_bin("infer-pro").unwrap();
    cmd.arg("--help").assert().success();
}

/// Basic smoke test for the `profile` subcommand.
#[test]
fn profile_subcommand_works() {
    let mut cmd = Command::cargo_bin("infer-pro").unwrap();
    cmd.arg("profile").assert().success();
}

/// Basic smoke test for the `info` subcommand.
#[test]
fn info_subcommand_works() {
    let mut cmd = Command::cargo_bin("infer-pro").unwrap();
    cmd.arg("info").assert().success();
}

