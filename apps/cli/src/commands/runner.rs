use anyhow::{Context, Result};
use tokio::sync::mpsc;
use uuid::Uuid;
use yaai_agent_loop::{AgentConfig, AgentRunner};
use yaai_llm::LlmClient;
use yaai_memory::SessionMemory;
use yaai_tools::{GrepFilesTool, ReadTool, ToolRegistry, ToolSchemaFormat};
use yaai_tracer::Tracer;

use super::llm::{build_llm_client, parse_provider_model, Provider};

pub const DEFAULT_SYSTEM_PROMPT: &str = "You are a helpful assistant with access to tools. \
    Use the `read` tool with an absolute path to read files from the filesystem.";
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
    token_tx: Option<mpsc::UnboundedSender<String>>,
) -> Result<(PromptRunResult, SessionMemory)> {
    let (provider, model) = parse_provider_model(&args.model)?;
    let llm = build_llm_client(&provider, &model)?;
    let tool_format = match provider {
        Provider::OpenAi => ToolSchemaFormat::OpenAi,
        Provider::Anthropic => ToolSchemaFormat::Anthropic,
    };
    run_prompt_with_client(
        prompt,
        args,
        llm.as_ref(),
        initial_memory,
        tool_format,
        token_tx,
    )
    .await
}

pub fn build_tool_registry() -> ToolRegistry {
    ToolRegistry::new()
        .register(ReadTool::new())
        .register(GrepFilesTool::new())
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
    tool_format: ToolSchemaFormat,
    token_tx: Option<mpsc::UnboundedSender<String>>,
) -> Result<(PromptRunResult, SessionMemory)> {
    let tools = build_tool_registry();

    let cwd = std::env::current_dir()
        .map(|p| p.display().to_string())
        .unwrap_or_default();
    let system_prompt = if cwd.is_empty() {
        DEFAULT_SYSTEM_PROMPT.to_string()
    } else {
        format!("{}\nWorking directory: {}", DEFAULT_SYSTEM_PROMPT, cwd)
    };
    let agent_config = AgentConfig {
        id: "prompt".to_string(),
        system_prompt,
        max_steps: DEFAULT_MAX_STEPS,
    };
    let tracer = Tracer::new(Uuid::new_v4(), &args.traces_dir)?;

    let runner = AgentRunner::new(&agent_config, llm, &tools, &tracer, tool_format)
        .with_memory(initial_memory);
    let runner = match token_tx {
        Some(tx) => runner.with_streaming(tx),
        None => runner,
    };
    let agent_result = runner.run(prompt).await;

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

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use yaai_llm::{LlmResponse, StubClient};

    #[test]
    fn build_tool_registry_includes_read_tool() {
        let registry = build_tool_registry();
        assert!(
            registry.names().contains(&"read"),
            "expected 'read' tool to be registered"
        );
    }

    #[test]
    fn build_tool_registry_includes_grep_files_tool() {
        let registry = build_tool_registry();
        assert!(
            registry.names().contains(&"grep_files"),
            "expected 'grep_files' tool to be registered"
        );
    }

    #[tokio::test]
    async fn run_prompt_with_client_returns_answer_and_steps() {
        let llm = StubClient::new(vec![LlmResponse::text("final answer")]);
        let traces = tempdir().unwrap();
        let args = ResolvedRunArgs {
            model: "openai/gpt-4o".to_string(),
            traces_dir: traces.path().display().to_string(),
        };

        let (result, _memory) = run_prompt_with_client(
            "hello",
            &args,
            &llm,
            SessionMemory::new(),
            ToolSchemaFormat::OpenAi,
            None,
        )
        .await
        .unwrap();

        assert_eq!(result.answer, "final answer");
        assert_eq!(result.steps_taken, 1);
    }

    #[tokio::test]
    async fn run_prompt_with_client_propagates_agent_error() {
        let llm = StubClient::new(vec![]);
        let traces = tempdir().unwrap();
        let args = ResolvedRunArgs {
            model: "openai/gpt-4o".to_string(),
            traces_dir: traces.path().display().to_string(),
        };

        let err = run_prompt_with_client(
            "hello",
            &args,
            &llm,
            SessionMemory::new(),
            ToolSchemaFormat::OpenAi,
            None,
        )
        .await
        .unwrap_err();

        assert!(
            err.to_string().contains("StubClient script exhausted"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn run_prompt_fails_on_invalid_provider_model() {
        let args = ResolvedRunArgs {
            model: "bogus/model".to_string(),
            traces_dir: tempdir().unwrap().path().display().to_string(),
        };
        let err = run_prompt("hi", &args, SessionMemory::new(), None)
            .await
            .unwrap_err();
        assert!(err.to_string().to_lowercase().contains("provider"));
    }
}
