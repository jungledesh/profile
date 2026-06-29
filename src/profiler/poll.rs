use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::time::Duration;

const POLL_INTERVAL: Duration = Duration::from_secs(2);
const WARMUP_DELAY: Duration = Duration::from_secs(5);
/// Per-request HTTP timeout for the restart poller. Prevents blocking indefinitely on a
/// slow-but-alive vLLM instance while waiting for restart detection.
const RESTART_POLL_REQUEST_TIMEOUT: Duration = Duration::from_secs(2);

pub enum WaitOutcome {
    RestartDetected,
    UserSkipped,
}

/// Spawns a single background thread that delivers one line per stdin read.
/// Call once before the loop and pass the receiver to `wait_for_restart_or_skip`
/// each iteration so stdin is never double-owned.
pub fn spawn_stdin_watcher() -> mpsc::Receiver<String> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        loop {
            let mut buf = String::new();
            if std::io::stdin().read_line(&mut buf).is_err() {
                return;
            }
            if tx.send(buf).is_err() {
                return;
            }
        }
    });
    rx
}

/// Waits until vLLM restarts (goes down then comes back up) OR user presses Enter.
/// `stdin_rx` must come from `spawn_stdin_watcher`.
pub fn wait_for_restart_or_skip(
    metrics_url: &str,
    stdin_rx: &mpsc::Receiver<String>,
) -> WaitOutcome {
    println!();
    println!("Press Enter when done.");

    let url = metrics_url.to_string();
    let (tx, rx) = mpsc::channel::<WaitOutcome>();
    let cancelled = Arc::new(AtomicBool::new(false));

    let cancelled_poller = Arc::clone(&cancelled);
    std::thread::spawn(move || {
        // Build once; reuse across poll iterations.
        let client = reqwest::blocking::Client::builder()
            .use_rustls_tls()
            .timeout(RESTART_POLL_REQUEST_TIMEOUT)
            .build()
            .ok();
        let mut was_down = false;
        loop {
            if cancelled_poller.load(Ordering::Relaxed) {
                return;
            }
            let reachable = client
                .as_ref()
                .and_then(|c| c.get(&url).send().ok())
                .map(|r| r.status().is_success())
                .unwrap_or(false);
            if !reachable && !was_down {
                was_down = true;
            }
            if reachable && was_down {
                println!(
                    "Connection restored. Resuming in {}s...",
                    WARMUP_DELAY.as_secs()
                );
                std::thread::sleep(WARMUP_DELAY);
                let _ = tx.send(WaitOutcome::RestartDetected);
                return;
            }
            std::thread::sleep(POLL_INTERVAL);
        }
    });

    // Discard any Enter presses that arrived while we weren't waiting.
    while stdin_rx.try_recv().is_ok() {}

    // Race: restart detection vs user pressing Enter.
    let outcome = loop {
        if stdin_rx.try_recv().is_ok() {
            break WaitOutcome::UserSkipped;
        }
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(o) => break o,
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => break WaitOutcome::UserSkipped,
        }
    };

    cancelled.store(true, Ordering::Relaxed);
    outcome
}
