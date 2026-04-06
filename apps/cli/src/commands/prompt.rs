use anyhow::Result;
use clap::Args;
use tracing::info;
use yaai_agent_loop::AgentConfig;
use yaai_orchestrator::run_single;
use yaai_tools::ToolRegistry;

use super::llm::{build_llm_client, parse_provider_model};

#[derive(Args)]
pub struct PromptArgs {
    #[arg(
        short = 'p',
        long,
        value_name = "PROMPT",
        help = "The prompt to send to the agent. The agent will reason over this input \
                and return a final answer, running up to a fixed number of steps."
    )]
    pub prompt: String,

    #[arg(
        short = 'm',
        long,
        value_name = "PROVIDER/MODEL",
        help = "The model to use, specified as provider/model (e.g. openai/gpt-4o, \
                anthropic/claude-3-5-sonnet-20241022). The corresponding API key must be \
                set in the environment (OPENAI_API_KEY or ANTHROPIC_API_KEY)."
    )]
    pub model: String,

    #[arg(
        long,
        default_value = "traces",
        help = "Directory where JSON trace files are written after each run. \
                Each run produces a file named <run-id>.json containing the full \
                step-by-step event log."
    )]
    pub traces_dir: String,
}

// grcov-excl-start
pub async fn execute(args: PromptArgs) -> Result<()> {
    let (provider, model) = parse_provider_model(&args.model)?;
    let llm = build_llm_client(&provider, &model)?;
    let tools = ToolRegistry::new();

    let agent_config = AgentConfig {
        id: "prompt".to_string(),
        system_prompt: "You are a helpful assistant.".to_string(),
        max_steps: 10,
    };

    let result = run_single(&agent_config, &args.prompt, &llm, &tools, &args.traces_dir).await?;

    info!(steps = result.steps_taken, "run complete");
    println!("{}", result.answer);

    Ok(())
}
// grcov-excl-stop
