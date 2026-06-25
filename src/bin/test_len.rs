use profile::output::stdout::*;
use owo_colors::OwoColorize;

fn main() {
    let s1 = "vLLM:".magenta().bold().to_string();
    let s2 = "ISSUES:".red().bold().to_string();
    println!("s1: len={}, visual={}", s1.len(), s1.chars().count());
    println!("s1={:?}", s1);
}
