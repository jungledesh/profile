//! Operator prompts injected into the diagnose loop.
//!
//! The loop never imports `cli/`. CLI implements this trait and passes it in.

use std::sync::mpsc;

/// Ask the operator for an updated `--max-num-seqs` after a fix iteration.
pub trait MaxNumSeqsPrompt {
    fn ask(&mut self, current: u32, stdin_rx: &mpsc::Receiver<String>) -> anyhow::Result<u32>;
}
