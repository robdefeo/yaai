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

    let result = run_single(&cfg("solo"), "do a task", &llm, &tools, dir.path().to_str().unwrap())
        .await
        .unwrap();

    assert_eq!(result.answer, "final answer");
    assert_eq!(result.agent_id, "solo");
}
