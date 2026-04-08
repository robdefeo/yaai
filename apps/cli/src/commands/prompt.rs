use anyhow::{bail, Result};
use clap::Args;
use tracing::info;
use yaai_agent_loop::AgentConfig;
use yaai_orchestrator::run_single;
use yaai_tools::ToolRegistry;

use crate::config::YaaiConfig;

use super::llm::{build_llm_client, parse_provider_model};

const DEFAULT_SYSTEM_PROMPT: &str = "You are a helpful assistant.";
const DEFAULT_MAX_STEPS: u32 = 10;
const DEFAULT_TRACES_DIR: &str = "traces";

#[derive(Args)]
pub struct PromptArgs {
    #[arg(
        short = 'p',
        long,
        value_name = "PROMPT",
        display_order = 1,
        help_heading = "Arguments",
        value_parser = |s: &str| -> Result<String, String> {
            if s.trim().is_empty() {
                Err("prompt must not be empty".to_string())
            } else {
                Ok(s.to_string())
            }
        },
        help = "The prompt to send to the agent. The agent will reason over this input \
                and return a final answer, running up to a fixed number of steps."
    )]
    pub prompt: String,

    #[arg(
        short = 'm',
        long,
        value_name = "PROVIDER/MODEL",
        display_order = 2,
        help_heading = "Arguments",
        help = "The model to use, specified as provider/model (e.g. openai/gpt-4o, \
                anthropic/claude-3-5-sonnet-20241022). The corresponding API key must be \
                set in the environment (OPENAI_API_KEY or ANTHROPIC_API_KEY). \
                Falls back to `model` in ~/.config/yaai/config.json if not set."
    )]
    pub model: Option<String>,

    #[arg(
        long,
        display_order = 13,
        help_heading = "Options",
        help = "Directory where trace files are written after each run. \
                Each run produces a file named <run-id>.ndjson containing \
                newline-delimited JSON (NDJSON) — one event object per line. \
                Falls back to `traces_dir` in ~/.config/yaai/config.json, then \"traces\"."
    )]
    pub traces_dir: Option<String>,
}

impl PromptArgs {
    /// Resolve final values by layering: CLI args > config file > hardcoded defaults.
    pub fn resolve(self, cfg: &YaaiConfig) -> Result<ResolvedPromptArgs> {
        let model = self.model.or_else(|| cfg.model.clone()).ok_or_else(|| {
            anyhow::anyhow!("--model is required (or set `model` in ~/.config/yaai/config.json)")
        })?;

        if model.trim().is_empty() {
            bail!("model must not be empty");
        }

        let traces_dir = self
            .traces_dir
            .or_else(|| cfg.traces_dir.clone())
            .unwrap_or_else(|| DEFAULT_TRACES_DIR.to_string());

        Ok(ResolvedPromptArgs {
            prompt: self.prompt,
            model,
            traces_dir,
        })
    }
}

#[derive(Debug)]
pub struct ResolvedPromptArgs {
    pub prompt: String,
    pub model: String,
    pub traces_dir: String,
}

// grcov-excl-start
pub async fn execute(args: PromptArgs, cfg: &YaaiConfig) -> Result<()> {
    let resolved = args.resolve(cfg)?;
    let (provider, model) = parse_provider_model(&resolved.model)?;
    let llm = build_llm_client(&provider, &model)?;
    let tools = ToolRegistry::new();

    let agent_config = AgentConfig {
        id: "prompt".to_string(),
        system_prompt: DEFAULT_SYSTEM_PROMPT.to_string(),
        max_steps: DEFAULT_MAX_STEPS,
    };

    let result = run_single(
        &agent_config,
        &resolved.prompt,
        &llm,
        &tools,
        &resolved.traces_dir,
    )
    .await?;

    info!(steps = result.steps_taken, "run complete");
    println!("{}", result.answer);

    Ok(())
}
// grcov-excl-stop

#[cfg(test)]
mod tests {
    use super::*;

    fn args(model: Option<&str>, traces_dir: Option<&str>) -> PromptArgs {
        PromptArgs {
            prompt: "hello".to_string(),
            model: model.map(str::to_string),
            traces_dir: traces_dir.map(str::to_string),
        }
    }

    fn cfg(model: Option<&str>, traces_dir: Option<&str>) -> YaaiConfig {
        YaaiConfig {
            model: model.map(str::to_string),
            traces_dir: traces_dir.map(str::to_string),
            json_logs: None,
        }
    }

    #[test]
    fn cli_model_takes_precedence_over_config() {
        let resolved = args(Some("openai/gpt-4o"), None)
            .resolve(&cfg(Some("anthropic/claude-3-5-sonnet-20241022"), None))
            .unwrap();
        assert_eq!(resolved.model, "openai/gpt-4o");
    }

    #[test]
    fn config_model_used_when_cli_model_absent() {
        let resolved = args(None, None)
            .resolve(&cfg(Some("anthropic/claude-3-5-sonnet-20241022"), None))
            .unwrap();
        assert_eq!(resolved.model, "anthropic/claude-3-5-sonnet-20241022");
    }

    #[test]
    fn missing_model_from_both_is_error() {
        let err = args(None, None).resolve(&cfg(None, None)).unwrap_err();
        assert!(err.to_string().contains("--model is required"));
    }

    #[test]
    fn cli_traces_dir_takes_precedence_over_config() {
        let resolved = args(Some("openai/gpt-4o"), Some("/cli/traces"))
            .resolve(&cfg(None, Some("/cfg/traces")))
            .unwrap();
        assert_eq!(resolved.traces_dir, "/cli/traces");
    }

    #[test]
    fn config_traces_dir_used_when_cli_absent() {
        let resolved = args(Some("openai/gpt-4o"), None)
            .resolve(&cfg(None, Some("/cfg/traces")))
            .unwrap();
        assert_eq!(resolved.traces_dir, "/cfg/traces");
    }

    #[test]
    fn default_traces_dir_when_both_absent() {
        let resolved = args(Some("openai/gpt-4o"), None)
            .resolve(&cfg(None, None))
            .unwrap();
        assert_eq!(resolved.traces_dir, DEFAULT_TRACES_DIR);
    }

    #[test]
    fn prompt_is_passed_through() {
        let resolved = args(Some("openai/gpt-4o"), None)
            .resolve(&cfg(None, None))
            .unwrap();
        assert_eq!(resolved.prompt, "hello");
    }
}
