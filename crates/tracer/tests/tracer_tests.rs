use std::path::PathBuf;

use uuid::Uuid;
use yaai_tracer::{init_tracing, EventKind, LogGuard, TraceEvent, Tracer};

fn parse_ndjson(content: &str) -> Vec<serde_json::Value> {
    content
        .lines()
        .filter(|l| !l.is_empty())
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

#[tokio::test]
async fn records_events_in_order() {
    let run_id = Uuid::new_v4();
    let tmp = tempfile::tempdir().unwrap();
    let dir: PathBuf = tmp.path().to_path_buf();
    let tracer = Tracer::new(run_id, &dir).unwrap();

    tracer
        .emit("agent-a", 0, EventKind::Prompt, "hello")
        .unwrap();
    tracer
        .emit("agent-a", 1, EventKind::ToolCall, "search")
        .unwrap();
    tracer
        .emit("agent-a", 2, EventKind::FinalAnswer, "done")
        .unwrap();

    tracer.close().await.unwrap();

    let content = tokio::fs::read_to_string(dir.join(format!("{run_id}.ndjson")))
        .await
        .unwrap();
    let events = parse_ndjson(&content);
    assert_eq!(events.len(), 3);
    assert_eq!(events[0]["kind"], "prompt");
    assert_eq!(events[1]["kind"], "tool_call");
    assert_eq!(events[2]["kind"], "final_answer");
    assert!(events.iter().all(|e| e["run_id"] == run_id.to_string()));
}

#[tokio::test]
async fn flush_writes_ndjson_file() {
    let run_id = Uuid::new_v4();
    let tmp = tempfile::tempdir().unwrap();
    let dir: PathBuf = tmp.path().to_path_buf();
    let tracer = Tracer::new(run_id, &dir).unwrap();

    tracer
        .emit("agent-a", 0, EventKind::FinalAnswer, "result")
        .unwrap();
    tracer.flush().await.unwrap();

    let content = tokio::fs::read_to_string(dir.join(format!("{run_id}.ndjson")))
        .await
        .unwrap();
    let events = parse_ndjson(&content);
    assert_eq!(events.len(), 1);
    assert_eq!(events[0]["kind"], "final_answer");

    tracer.close().await.unwrap();
}

#[tokio::test]
async fn events_visible_before_close() {
    // Verifies that flush() makes events readable without closing the tracer,
    // which is the key property for tailing a live run.
    let run_id = Uuid::new_v4();
    let tmp = tempfile::tempdir().unwrap();
    let dir: PathBuf = tmp.path().to_path_buf();
    let tracer = Tracer::new(run_id, &dir).unwrap();

    tracer
        .emit("agent-a", 0, EventKind::Prompt, "step one")
        .unwrap();
    tracer.flush().await.unwrap();

    let content = tokio::fs::read_to_string(dir.join(format!("{run_id}.ndjson")))
        .await
        .unwrap();
    assert_eq!(parse_ndjson(&content).len(), 1);

    tracer
        .emit("agent-a", 1, EventKind::FinalAnswer, "step two")
        .unwrap();
    tracer.close().await.unwrap();

    let content = tokio::fs::read_to_string(dir.join(format!("{run_id}.ndjson")))
        .await
        .unwrap();
    assert_eq!(parse_ndjson(&content).len(), 2);
}

#[tokio::test]
async fn run_id_is_stable() {
    let id = Uuid::new_v4();
    let tmp = tempfile::tempdir().unwrap();
    let tracer = Tracer::new(id, tmp.path()).unwrap();
    assert_eq!(tracer.run_id(), id);
    tracer.close().await.unwrap();
}

#[test]
fn all_event_kinds_serialize() {
    let cases = [
        (EventKind::Prompt, "prompt"),
        (EventKind::ToolCall, "tool_call"),
        (EventKind::ToolResult, "tool_result"),
        (EventKind::Decision, "decision"),
        (EventKind::FinalAnswer, "final_answer"),
        (EventKind::Error, "error"),
    ];
    for (kind, expected) in cases {
        let json = serde_json::to_string(&kind).unwrap();
        assert_eq!(json, format!("\"{}\"", expected), "kind {:?} wrong", kind);
    }
}

#[test]
fn trace_event_new_fields() {
    let run_id = Uuid::new_v4();
    let event = TraceEvent::new(run_id, "my-agent", 3, EventKind::Decision, "thinking").unwrap();
    assert_eq!(event.run_id, run_id);
    assert_eq!(event.agent_id, "my-agent");
    assert_eq!(event.step, 3);
    assert_eq!(event.kind, EventKind::Decision);
    assert_eq!(event.payload, serde_json::json!("thinking"));
}

#[tokio::test]
async fn record_directly() {
    let run_id = Uuid::new_v4();
    let tmp = tempfile::tempdir().unwrap();
    let dir: PathBuf = tmp.path().to_path_buf();
    let tracer = Tracer::new(run_id, &dir).unwrap();

    let event = TraceEvent::new(run_id, "agent-x", 0, EventKind::ToolResult, "result").unwrap();
    tracer.record(event);
    tracer.close().await.unwrap();

    let content = tokio::fs::read_to_string(dir.join(format!("{run_id}.ndjson")))
        .await
        .unwrap();
    let events = parse_ndjson(&content);
    assert_eq!(events.len(), 1);
    assert_eq!(events[0]["kind"], "tool_result");
}

#[test]
fn tracer_new_fails_on_uncreatable_dir() {
    let tmp = tempfile::tempdir().unwrap();
    let file = tmp.path().join("file.txt");
    std::fs::write(&file, "").unwrap();
    // A file cannot be used as a directory — create_dir_all should fail.
    let bad_dir = file.join("subdir");
    let result = Tracer::new(Uuid::new_v4(), &bad_dir);
    assert!(result.is_err());
}

#[test]
fn init_tracing_falls_back_to_noop_on_bad_dir() {
    let tmp = tempfile::tempdir().unwrap();
    let file = tmp.path().join("file.txt");
    std::fs::write(&file, "").unwrap();
    // Appender will fail because `file.txt` is a file, not a directory.
    let bad_dir = file.join("logs");
    let guard = init_tracing(false, &bad_dir);
    assert!(matches!(guard, LogGuard::Noop));
}

#[test]
fn init_tracing_json_falls_back_to_noop_on_bad_dir() {
    let tmp = tempfile::tempdir().unwrap();
    let file = tmp.path().join("file2.txt");
    std::fs::write(&file, "").unwrap();
    let bad_dir = file.join("logs");
    let guard = init_tracing(true, &bad_dir);
    assert!(matches!(guard, LogGuard::Noop));
}
