use yaai_agent_loop::AgentConfig;
use yaai_llm::{LlmResponse, StubClient};
use yaai_orchestrator::run_single;
use yaai_tools::ToolRegistry;

fn cfg(id: &str) -> AgentConfig {
    AgentConfig {
        id: id.to_string(),
        system_prompt: format!("You are the {id} agent."),
        max_steps: 10,
    }
}

#[tokio::test]
async fn single_agent_completes() {
    let llm = StubClient::new(vec![LlmResponse::text("final answer")]);
    let tools = ToolRegistry::new();
    let dir = tempfile::tempdir().unwrap();

    let result = run_single(
        &cfg("solo"),
        "do a task",
        &llm,
        &tools,
        dir.path().to_str().unwrap(),
    )
    .await
    .unwrap();

    assert_eq!(result.answer, "final answer");
    assert_eq!(result.agent_id, "solo");
}

#[tokio::test]
async fn single_agent_closes_tracer_on_error() {
    let llm = StubClient::new(vec![]);
    let tools = ToolRegistry::new();
    let dir = tempfile::tempdir().unwrap();

    let err = run_single(
        &cfg("solo"),
        "do a task",
        &llm,
        &tools,
        dir.path().to_str().unwrap(),
    )
    .await
    .unwrap_err();

    assert!(
        err.to_string().contains("StubClient script exhausted"),
        "unexpected error: {err}"
    );

    let trace_files = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect::<Vec<_>>();

    assert_eq!(trace_files.len(), 1);
    assert_eq!(
        trace_files[0].extension().and_then(|ext| ext.to_str()),
        Some("ndjson")
    );
}
