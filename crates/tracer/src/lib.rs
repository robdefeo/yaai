//! Structured tracing for agent runs.
//!
//! Each agent run emits a sequence of [`TraceEvent`]s written to
//! `<output_dir>/<run_id>.ndjson` as newline-delimited JSON. Events are
//! written asynchronously via a background task, so the file can be tailed
//! while the agent is running. Call [`Tracer::flush`] to wait for all queued
//! events to be written.

use std::path::PathBuf;

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::io::AsyncWriteExt;
use tokio::sync::{mpsc, oneshot};
use uuid::Uuid;

/// The kind of event recorded in a trace.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventKind {
    /// The prompt sent to the LLM at a given step.
    Prompt,
    /// The LLM decided to call a tool.
    ToolCall,
    /// The tool returned a result.
    ToolResult,
    /// The LLM produced a reasoning/decision step.
    Decision,
    /// The agent produced a final answer and the loop ended.
    FinalAnswer,
    /// An error occurred (tool failure, LLM error, etc.).
    Error,
}

/// A single event in an agent run trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceEvent {
    pub run_id: Uuid,
    pub agent_id: String,
    pub step: u32,
    pub kind: EventKind,
    pub payload: serde_json::Value,
    pub timestamp: DateTime<Utc>,
}

impl TraceEvent {
    pub fn new(
        run_id: Uuid,
        agent_id: impl Into<String>,
        step: u32,
        kind: EventKind,
        payload: impl Serialize,
    ) -> Result<Self> {
        Ok(Self {
            run_id,
            agent_id: agent_id.into(),
            step,
            kind,
            payload: serde_json::to_value(payload)?,
            timestamp: Utc::now(),
        })
    }
}

enum WriterMsg {
    Event(TraceEvent),
    /// Flush OS buffers; ack when done.
    Flush(oneshot::Sender<()>),
}

/// Streams trace events to `<output_dir>/<run_id>.ndjson` as they are recorded.
///
/// Events are sent to a background writer task over an in-process channel, so
/// `record` and `emit` are non-blocking. Call `flush` to wait for pending writes
/// to reach disk, and `close` to shut down cleanly at the end of a run.
#[derive(Debug)]
pub struct Tracer {
    run_id: Uuid,
    tx: mpsc::UnboundedSender<WriterMsg>,
    handle: tokio::task::JoinHandle<Result<()>>,
}

impl Tracer {
    /// Create a new tracer and start the background writer task.
    ///
    /// Events are written to `<output_dir>/<run_id>.ndjson`. The directory is
    /// created if it does not exist.
    ///
    /// # Panics
    ///
    /// Panics if called outside of a Tokio runtime context.
    pub fn new(run_id: Uuid, output_dir: impl Into<PathBuf>) -> Result<Self> {
        let output_dir = output_dir.into();
        std::fs::create_dir_all(&output_dir).context("creating traces output dir")?;

        let path = output_dir.join(format!("{run_id}.ndjson"));
        let (tx, rx) = mpsc::unbounded_channel::<WriterMsg>();

        let handle = tokio::spawn(writer_task(path, rx));

        Ok(Self { run_id, tx, handle })
    }

    /// Record an event. Non-blocking — the event is queued for the writer task.
    pub fn record(&self, event: TraceEvent) {
        tracing::debug!(
            run_id = %event.run_id,
            step = event.step,
            kind = ?event.kind,
            "trace event"
        );
        // Only fails if the writer task has exited (e.g. panicked).
        if let Err(err) = self.tx.send(WriterMsg::Event(event)) {
            tracing::error!( // grcov-excl-line
                run_id = %self.run_id, // grcov-excl-line
                error = ?err, // grcov-excl-line
                "tracer writer task has exited; dropping trace event" // grcov-excl-line
            ); // grcov-excl-line
        }
    }

    /// Convenience: build and record an event for this run.
    pub fn emit(
        &self,
        agent_id: impl Into<String>,
        step: u32,
        kind: EventKind,
        payload: impl Serialize,
    ) -> Result<()> {
        let event = TraceEvent::new(self.run_id, agent_id, step, kind, payload)?;
        self.record(event);
        Ok(())
    }

    /// The run ID for this tracer.
    pub fn run_id(&self) -> Uuid {
        self.run_id
    }

    /// Wait until all queued events have been written to the OS write buffer.
    ///
    /// This is a write-barrier: it guarantees that all events emitted before
    /// this call have been processed by the writer task and handed to the OS.
    /// It does **not** guarantee durability (no `fsync`/`sync_data` is issued).
    pub async fn flush(&self) -> Result<()> {
        let (ack_tx, ack_rx) = oneshot::channel();
        let _ = self.tx.send(WriterMsg::Flush(ack_tx));
        ack_rx
            .await
            .context("tracer writer task exited before flush")
    }

    /// Flush all pending events and shut down the writer task.
    ///
    /// Must be called at the end of a run to ensure all events reach disk.
    pub async fn close(self) -> Result<()> {
        drop(self.tx);
        self.handle.await.context("tracer writer task panicked")?
    }
}

async fn writer_task(path: PathBuf, mut rx: mpsc::UnboundedReceiver<WriterMsg>) -> Result<()> {
    let mut file = tokio::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .await
        .with_context(|| format!("opening trace file {}", path.display()))?;

    while let Some(msg) = rx.recv().await {
        match msg {
            WriterMsg::Event(event) => {
                let mut line = serde_json::to_string(&event).context("serialising trace event")?;
                line.push('\n');
                file.write_all(line.as_bytes())
                    .await
                    .with_context(|| format!("writing to {}", path.display()))?;
            }
            WriterMsg::Flush(ack) => {
                file.flush()
                    .await
                    .with_context(|| format!("flushing {}", path.display()))?;
                let _ = ack.send(());
            }
        }
    }

    // Sender dropped (close() called) — flush and exit cleanly.
    file.flush().await.context("final flush")?;
    tracing::info!(path = %path.display(), "trace closed");
    Ok(())
}

/// Holds the active log writer for the lifetime of the process.
///
/// Returned by [`init_tracing`] and must be kept alive until the process exits. When dropped,
/// the background log-writer thread is flushed and shut down.
pub enum LogGuard {
    /// File logging is active; dropping this flushes the writer thread.
    File(tracing_appender::non_blocking::WorkerGuard),
    /// File logging could not be initialised; events are discarded.
    Noop,
}

/// Initialise a `tracing-subscriber` for the process, writing logs to rolling daily files
/// under `log_dir`. Files are named `yaai.YYYY-MM-DD.log` and the seven most recent are kept.
///
/// File logging is **best-effort**: if the log directory cannot be created or the appender
/// cannot be initialised, a no-op subscriber is installed instead and the function still
/// returns successfully. This ensures a logging failure never prevents the CLI from running.
///
/// The returned [`LogGuard`] **must be held** for the lifetime of the process.
pub fn init_tracing(json: bool, log_dir: &std::path::Path) -> LogGuard {
    use tracing_subscriber::{fmt, EnvFilter};

    let filter = || EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    match try_file_appender(log_dir) {
        Ok(appender) => {
            let (non_blocking, guard) = tracing_appender::non_blocking(appender);
            let result = if json {
                fmt()
                    .json()
                    .with_env_filter(filter())
                    .with_writer(non_blocking)
                    .try_init()
            } else {
                fmt()
                    .with_env_filter(filter())
                    .with_writer(non_blocking)
                    .try_init()
            };
            if result.is_ok() {
                return LogGuard::File(guard);
            }
            // Subscriber already set (e.g. in tests); guard is dropped, no file logging.
        }
        Err(_) => {
            // Appender failed; install a sink subscriber so events are silently discarded.
            let _ = fmt().with_writer(std::io::sink).try_init();
        }
    }

    LogGuard::Noop
}

fn try_file_appender(
    log_dir: &std::path::Path,
) -> Result<tracing_appender::rolling::RollingFileAppender> {
    use tracing_appender::rolling::{RollingFileAppender, Rotation};

    std::fs::create_dir_all(log_dir)
        .with_context(|| format!("creating log directory {}", log_dir.display()))?;

    RollingFileAppender::builder()
        .rotation(Rotation::DAILY)
        .filename_prefix("yaai")
        .filename_suffix("log")
        .max_log_files(7)
        .build(log_dir)
        .with_context(|| format!("initialising log appender in {}", log_dir.display()))
}
