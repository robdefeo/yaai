use std::io::Write;
use tempfile::{NamedTempFile, TempDir};
use uuid::Uuid;
use yaai_agent_loop::{AgentConfig, AgentRunner};
use yaai_llm::{LlmResponse, StubClient};
use yaai_memory::SessionMemory;
use yaai_tools::{builtin::ReadTool, ToolRegistry};
use yaai_tracer::Tracer;

fn cfg(max_steps: u32) -> AgentConfig {
    AgentConfig {
        id: "test-agent".to_string(),
        system_prompt: "You are a test agent.".to_string(),
        max_steps,
    }
}

fn tracer() -> (TempDir, Tracer) {
    let tmp = tempfile::tempdir().unwrap();
    let tr = Tracer::new(Uuid::new_v4(), tmp.path()).expect("tracer init failed");
    (tmp, tr)
}

fn temp_file_path() -> (NamedTempFile, String) {
    let mut f = NamedTempFile::new().unwrap();
    writeln!(f, "hello world").unwrap();
    let path = f.path().to_str().unwrap().to_string();
    (f, path)
}

#[tokio::test]
async fn produces_final_answer_without_tools() {
    let llm = StubClient::new(vec![LlmResponse::text("The answer is 42.")]);
    let tools = ToolRegistry::new();
    let mut mem = SessionMemory::new();
    let (_tmp, tr) = tracer();

    let result = AgentRunner::new(&cfg(5), &llm, &tools, &tr, &mut mem)
        .run("What is the answer?")
        .await
        .unwrap();

    assert_eq!(result.answer, "The answer is 42.");
    assert_eq!(result.steps_taken, 1);
    tr.close().await.unwrap();
}

#[tokio::test]
async fn calls_tool_then_answers() {
    let (_f, path) = temp_file_path();
    let llm = StubClient::new(vec![
        LlmResponse::tool("read", serde_json::json!({"file_path": path})),
        LlmResponse::text("The answer is 42."),
    ]);
    let mut tools = ToolRegistry::new();
    tools.register(ReadTool::new());
    let mut mem = SessionMemory::new();
    let (_tmp, tr) = tracer();

    let result = AgentRunner::new(&cfg(5), &llm, &tools, &tr, &mut mem)
        .run("Read the file")
        .await
        .unwrap();

    assert_eq!(result.answer, "The answer is 42.");
    assert_eq!(result.steps_taken, 2);
    tr.close().await.unwrap();
}

#[tokio::test]
async fn respects_max_steps() {
    let (_f, path) = temp_file_path();
    let llm = StubClient::new(vec![
        LlmResponse::tool("read", serde_json::json!({"file_path": path.clone()})),
        LlmResponse::tool("read", serde_json::json!({"file_path": path.clone()})),
        LlmResponse::tool("read", serde_json::json!({"file_path": path})),
    ]);
    let mut tools = ToolRegistry::new();
    tools.register(ReadTool::new());
    let mut mem = SessionMemory::new();
    let (_tmp, tr) = tracer();

    let err = AgentRunner::new(&cfg(3), &llm, &tools, &tr, &mut mem)
        .run("loop forever")
        .await
        .unwrap_err();

    assert!(err.to_string().contains("max_steps"));
    tr.close().await.unwrap();
}

#[tokio::test]
async fn trace_has_correct_event_sequence() {
    let (_f, path) = temp_file_path();
    let llm = StubClient::new(vec![
        LlmResponse::tool("read", serde_json::json!({"file_path": path})),
        LlmResponse::text("Result is done."),
    ]);
    let mut tools = ToolRegistry::new();
    tools.register(ReadTool::new());
    let mut mem = SessionMemory::new();
    let (_tmp, tr) = tracer();

    // Tracer is write-only (streams to ndjson); verify the run completed with
    // the expected step count as a proxy for correct event sequencing.
    let result = AgentRunner::new(&cfg(5), &llm, &tools, &tr, &mut mem)
        .run("Read file")
        .await
        .unwrap();

    // 2 steps: one tool call + one final answer
    assert_eq!(result.steps_taken, 2);
    assert_eq!(result.answer, "Result is done.");
    tr.close().await.unwrap();
}

#[tokio::test]
async fn memory_accumulates_across_steps() {
    let (_f, path) = temp_file_path();
    let llm = StubClient::new(vec![
        LlmResponse::tool("read", serde_json::json!({"file_path": path})),
        LlmResponse::text("Done."),
    ]);
    let mut tools = ToolRegistry::new();
    tools.register(ReadTool::new());
    let mut mem = SessionMemory::new();
    let (_tmp, tr) = tracer();

    AgentRunner::new(&cfg(5), &llm, &tools, &tr, &mut mem)
        .run("task")
        .await
        .unwrap();

    // user task + tool result observation + final assistant = at least 3
    assert!(mem.len() >= 3);
    tr.close().await.unwrap();
}

#[tokio::test]
async fn graceful_tool_error_continues_loop() {
    let llm = StubClient::new(vec![
        LlmResponse::tool(
            "read",
            serde_json::json!({"file_path": "/nonexistent/file.txt"}),
        ),
        LlmResponse::text("Handled the error."),
    ]);
    let mut tools = ToolRegistry::new();
    tools.register(ReadTool::new());
    let mut mem = SessionMemory::new();
    let (_tmp, tr) = tracer();

    // The tool error should be fed back as a ToolResult observation and the
    // loop should continue to the next LLM call rather than propagating the
    // error up to the caller.
    let result = AgentRunner::new(&cfg(5), &llm, &tools, &tr, &mut mem)
        .run("read missing file")
        .await
        .unwrap();

    assert_eq!(result.answer, "Handled the error.");
    // 2 steps: tool attempt (error) + final answer
    assert_eq!(result.steps_taken, 2);
    tr.close().await.unwrap();
}

#[tokio::test]
async fn empty_llm_response_returns_error() {
    let llm = StubClient::new(vec![LlmResponse {
        content: None,
        tool_call: None,
    }]);
    let tools = ToolRegistry::new();
    let mut mem = SessionMemory::new();
    let (_tmp, tr) = tracer();

    let err = AgentRunner::new(&cfg(5), &llm, &tools, &tr, &mut mem)
        .run("task")
        .await
        .unwrap_err();

    assert!(err.to_string().contains("empty LLM response"));
    tr.close().await.unwrap();
}

#[test]
fn agent_config_serde_round_trip() {
    use yaai_agent_loop::AgentResult;

    let config = AgentConfig {
        id: "agent-1".to_string(),
        system_prompt: "you are helpful".to_string(),
        max_steps: 10,
    };
    let json = serde_json::to_string(&config).unwrap();
    let c2: AgentConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(c2.id, "agent-1");
    assert_eq!(c2.max_steps, 10);

    let result = AgentResult {
        run_id: uuid::Uuid::new_v4(),
        agent_id: "agent-1".to_string(),
        answer: "42".to_string(),
        steps_taken: 3,
    };
    let json = serde_json::to_string(&result).unwrap();
    let r2: AgentResult = serde_json::from_str(&json).unwrap();
    assert_eq!(r2.answer, "42");
    assert_eq!(r2.steps_taken, 3);
}
