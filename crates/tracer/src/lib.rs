//! Structured tracing for agent runs.
//!
//! Each agent run emits a sequence of [`TraceEvent`]s written to
//! `<output_dir>/<run_id>.ndjson` as newline-delimited JSON. Events are
//! written to disk immediately as they are recorded, so the file can be
//! tailed while the agent is running.

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
        let _ = self.tx.send(WriterMsg::Event(event));
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

    /// Wait until all queued events have been written and flushed to the OS.
    pub async fn flush(&self) -> Result<()> {
        let (ack_tx, ack_rx) = oneshot::channel();
        let _ = self.tx.send(WriterMsg::Flush(ack_tx));
        ack_rx.await.context("tracer writer task exited before flush")
    }

    /// Flush all pending events and shut down the writer task.
    ///
    /// Must be called at the end of a run to ensure all events reach disk.
    pub async fn close(self) -> Result<()> {
        drop(self.tx);
        self.handle
            .await
            .context("tracer writer task panicked")?
    }
}

async fn writer_task(
    path: PathBuf,
    mut rx: mpsc::UnboundedReceiver<WriterMsg>,
) -> Result<()> {
    let mut file = tokio::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .await
        .with_context(|| format!("opening trace file {}", path.display()))?;

    while let Some(msg) = rx.recv().await {
        match msg {
            WriterMsg::Event(event) => {
                let mut line = serde_json::to_string(&event)
                    .context("serialising trace event")?;
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

/// Initialise a `tracing-subscriber` for the process (JSON or pretty).
pub fn init_tracing(json: bool) {
    use tracing_subscriber::{fmt, EnvFilter};

    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    if json {
        fmt().json().with_env_filter(filter).init();
    } else {
        fmt().pretty().with_env_filter(filter).init();
    }
}
