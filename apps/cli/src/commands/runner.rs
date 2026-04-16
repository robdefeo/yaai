use anyhow::{Context, Result};
use uuid::Uuid;
use yaai_agent_loop::{AgentConfig, AgentRunner};
use yaai_llm::LlmClient;
use yaai_memory::SessionMemory;
use yaai_tools::ToolRegistry;
use yaai_tracer::Tracer;

use super::llm::{build_llm_client, parse_provider_model};

pub const DEFAULT_SYSTEM_PROMPT: &str = "You are a helpful assistant.";
pub const DEFAULT_MAX_STEPS: u32 = 10;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedRunArgs {
    pub model: String,
    pub traces_dir: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptRunResult {
    pub answer: String,
    pub steps_taken: u32,
}

pub async fn run_prompt(
    prompt: &str,
    args: &ResolvedRunArgs,
    initial_memory: SessionMemory,
) -> Result<(PromptRunResult, SessionMemory)> {
    let (provider, model) = parse_provider_model(&args.model)?;
    let llm = build_llm_client(&provider, &model)?;
    run_prompt_with_client(prompt, args, llm.as_ref(), initial_memory).await
}

/// Run a prompt, returning the result and the updated conversation history.
///
/// Pass a fresh [`SessionMemory`] for a stateless run, or a snapshot from a
/// previous result for multi-turn conversation.
pub async fn run_prompt_with_client(
    prompt: &str,
    args: &ResolvedRunArgs,
    llm: &dyn LlmClient,
    initial_memory: SessionMemory,
) -> Result<(PromptRunResult, SessionMemory)> {
    let tools = ToolRegistry::new();
    let agent_config = AgentConfig {
        id: "prompt".to_string(),
        system_prompt: DEFAULT_SYSTEM_PROMPT.to_string(),
        max_steps: DEFAULT_MAX_STEPS,
    };
    let tracer = Tracer::new(Uuid::new_v4(), &args.traces_dir)?;

    let agent_result = AgentRunner::new(&agent_config, llm, &tools, &tracer)
        .with_memory(initial_memory)
        .run(prompt)
        .await;

    let close_result = tracer.close().await;

    let agent_result = match (agent_result, close_result) {
        (Ok(r), Ok(())) => Ok(r),
        (Err(run_err), Ok(())) => Err(run_err),
        (Ok(_), Err(close_err)) => Err(close_err),
        (Err(run_err), Err(close_err)) => {
            Err(run_err).context(format!("failed to close tracer cleanly: {close_err}"))
        }
    }?;

    Ok((
        PromptRunResult {
            answer: agent_result.answer,
            steps_taken: agent_result.steps_taken,
        },
        agent_result.memory,
    ))
}

// grcov-excl-start: exclude inline unit tests from production coverage
#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use yaai_llm::{LlmResponse, StubClient};

    #[tokio::test]
    async fn run_prompt_with_client_returns_answer_and_steps() {
        let llm = StubClient::new(vec![LlmResponse::text("final answer")]);
        let traces = tempdir().unwrap();
        let args = ResolvedRunArgs {
            model: "openai/gpt-4o".to_string(),
            traces_dir: traces.path().display().to_string(),
        };

        let (result, _memory) = run_prompt_with_client("hello", &args, &llm, SessionMemory::new())
            .await
            .unwrap();

        assert_eq!(result.answer, "final answer");
        assert_eq!(result.steps_taken, 1);
    }
}
// grcov-excl-stop
