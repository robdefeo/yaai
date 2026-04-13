use anyhow::Result;
use yaai_agent_loop::AgentConfig;
use yaai_llm::LlmClient;
use yaai_orchestrator::run_single;
use yaai_tools::ToolRegistry;

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

pub async fn run_prompt(prompt: &str, args: &ResolvedRunArgs) -> Result<PromptRunResult> {
    let (provider, model) = parse_provider_model(&args.model)?;
    let llm = build_llm_client(&provider, &model)?;
    run_prompt_with_client(prompt, args, llm.as_ref()).await
}

pub async fn run_prompt_with_client(
    prompt: &str,
    args: &ResolvedRunArgs,
    llm: &dyn LlmClient,
) -> Result<PromptRunResult> {
    let tools = ToolRegistry::new();
    let agent_config = AgentConfig {
        id: "prompt".to_string(),
        system_prompt: DEFAULT_SYSTEM_PROMPT.to_string(),
        max_steps: DEFAULT_MAX_STEPS,
    };

    let result = run_single(&agent_config, prompt, llm, &tools, &args.traces_dir).await?;

    Ok(PromptRunResult {
        answer: result.answer,
        steps_taken: result.steps_taken,
    })
}

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

        let result = run_prompt_with_client("hello", &args, &llm).await.unwrap();

        assert_eq!(result.answer, "final answer");
        assert_eq!(result.steps_taken, 1);
    }
}
