use anyhow::{Context, Result};
use uuid::Uuid;
use yaai_agent_loop::{AgentConfig, AgentResult, AgentRunner};
use yaai_llm::LlmClient;
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

    let result = AgentRunner::new(config, llm, tools, &tracer)
        .run(task)
        .await;

    let close_result = tracer.close().await;

    match (result, close_result) {
        (Ok(result), Ok(())) => Ok(result),
        (Err(run_err), Ok(())) => Err(run_err),
        (Ok(_), Err(close_err)) => Err(close_err),
        (Err(run_err), Err(close_err)) => {
            Err(run_err).context(format!("failed to close tracer cleanly: {close_err}"))
        }
    }
}
