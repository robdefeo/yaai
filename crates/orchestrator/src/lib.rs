//! Agent orchestration — single-agent and multi-agent sequential workflows.

use anyhow::Result;
use tracing::info;
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
    run_id: Uuid,
    traces_dir: &str,
) -> Result<AgentResult> {
    let tracer = Tracer::new(run_id, traces_dir)?;
    let mut memory = SessionMemory::new();

    let result = AgentRunner::new(config, llm, tools, &tracer, &mut memory)
        .run(task)
        .await?;

    tracer.close().await?;
    Ok(result)
}

/// One step in a sequential multi-agent workflow.
#[derive(Debug, Clone)]
pub struct WorkflowStep {
    pub config: AgentConfig,
}

/// Run a sequential multi-agent workflow.
///
/// Each agent's answer becomes the next agent's task input.
/// All agents share one `run_id` so their events appear in a single trace file.
pub async fn run_sequential(
    steps: &[WorkflowStep],
    initial_task: impl Into<String>,
    llm: &dyn LlmClient,
    tools: &ToolRegistry,
    traces_dir: &str,
) -> Result<Vec<AgentResult>> {
    let run_id = Uuid::new_v4();
    let tracer = Tracer::new(run_id, traces_dir)?;

    let mut current_task = initial_task.into();
    let mut results = Vec::new();

    for step in steps {
        let mut memory = SessionMemory::new();
        let result = AgentRunner::new(&step.config, llm, tools, &tracer, &mut memory)
            .run(&current_task)
            .await?;

        info!(agent = %result.agent_id, steps = result.steps_taken, "agent completed");
        current_task = result.answer.clone();
        results.push(result);
    }

    tracer.close().await?;
    Ok(results)
}
