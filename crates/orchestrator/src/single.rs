use anyhow::Result;
use uuid::Uuid;
use yaai_agent_loop::{AgentConfig, AgentResult, AgentRunner};
use yaai_llm::LlmClient;
use yaai_memory::SessionMemory;
use yaai_tools::ToolRegistry;
use yaai_tracer::Tracer;

/// Run a single agent on a task, flush the trace, and return the result.
pub async fn run_single(
    config: &AgentConfig,
    task: impl Into<String>,
    llm: &dyn LlmClient,
    tools: &ToolRegistry,
    traces_dir: &str,
) -> Result<AgentResult> {
    let tracer = Tracer::new(Uuid::new_v4(), traces_dir)?;
    let mut memory = SessionMemory::new();

    let result = AgentRunner::new(config, llm, tools, &tracer, &mut memory)
        .run(task)
        .await?;

    tracer.close().await?;
    Ok(result)
}
