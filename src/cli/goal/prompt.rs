use std::io::Write;

use super::types::{Goal, InferredObjective, Objective};

const OPTIONS: &[Objective] = &[
    Objective::MaxThroughput,
    Objective::MinLatency,
    Objective::ReduceCost,
];

pub fn prompt_goal(inferred: &InferredObjective) -> std::io::Result<Goal> {
    println!();
    println!("Profile found: {}", inferred.reason);
    println!();
    println!("Goal:");
    for (i, opt) in OPTIONS.iter().enumerate() {
        let tag = if *opt == inferred.objective {
            "  ← recommended"
        } else {
            ""
        };
        println!("  [{}] {}{}", i + 1, opt.label(), tag);
    }
    let default_pos = OPTIONS
        .iter()
        .position(|o| *o == inferred.objective)
        .unwrap_or(0);
    print!(
        "\nEnter 1–3, or press Enter to accept [{}]: ",
        default_pos + 1
    );
    std::io::stdout().flush()?;

    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;

    let objective = match input.trim() {
        "" => inferred.objective,
        "1" => Objective::MaxThroughput,
        "2" => Objective::MinLatency,
        "3" => Objective::ReduceCost,
        _ => {
            println!("Unrecognised — using: {}", inferred.objective.label());
            inferred.objective
        }
    };
    Ok(Goal { objective })
}
