use anyhow::{bail, Result};
use clap::Args;
use tracing::info;

use crate::config::{self, YaaiConfig};

use super::runner::{run_prompt, ResolvedRunArgs};
use yaai_memory::SessionMemory;

fn expand_tilde(p: String) -> String {
    if let Some(rest) = p.strip_prefix("~/") {
        if let Some(home) = dirs::home_dir() {
            return home.join(rest).to_string_lossy().into_owned();
        }
    }
    p
}

const DEFAULT_TRACES_DIR: &str = "traces";

#[derive(Args, Debug, Clone)]
pub struct PromptArgs {
    #[arg(
        short = 'p',
        long,
        value_name = "PROMPT",
        display_order = 1,
        help_heading = "Arguments",
        help = "Optional prompt to run non-interactively. When omitted, `yaai` starts the TUI."
    )]
    pub prompt: Option<String>,

    #[arg(
        short = 'm',
        long,
        value_name = "PROVIDER/MODEL",
        display_order = 2,
        help_heading = "Arguments",
        help = "The model to use, specified as provider/model (e.g. openai/gpt-4o, \
                anthropic/claude-3-5-sonnet-20241022). The corresponding API key must be \
                set in the environment (OPENAI_API_KEY or ANTHROPIC_API_KEY)."
    )]
    pub model: Option<String>,

    #[arg(
        long,
        display_order = 13,
        help_heading = "Options",
        help = "Directory where trace files are written after each run. \
                Each run produces a file named <run-id>.ndjson containing \
                newline-delimited JSON (NDJSON) — one event object per line."
    )]
    pub traces_dir: Option<String>,
}

impl PromptArgs {
    /// Resolve final values by layering: CLI args > config file > hardcoded defaults.
    pub fn resolve_run_args(&self, cfg: &YaaiConfig) -> Result<ResolvedRunArgs> {
        let model = self
            .model
            .clone()
            .or_else(|| cfg.model.clone())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "--model is required (or set `model` in {})",
                    config::config_path_display()
                )
            })?;

        if model.trim().is_empty() {
            bail!("model must not be empty");
        }

        let traces_dir = self
            .traces_dir
            .clone()
            .or_else(|| cfg.traces_dir.clone())
            .unwrap_or_else(|| DEFAULT_TRACES_DIR.to_string());
        let traces_dir = expand_tilde(traces_dir);

        Ok(ResolvedRunArgs { model, traces_dir })
    }

    pub fn prompt_text(&self) -> Result<Option<String>> {
        match self.prompt.as_deref() {
            Some(prompt) if prompt.trim().is_empty() => bail!("prompt must not be empty"),
            Some(prompt) => Ok(Some(prompt.to_string())),
            None => Ok(None),
        }
    }
}

// grcov-excl-start: non-interactive CLI output is thin stdout wiring
pub async fn execute_non_interactive(args: &PromptArgs, cfg: &YaaiConfig) -> Result<()> {
    let prompt = args
        .prompt_text()?
        .ok_or_else(|| anyhow::anyhow!("prompt is required for non-interactive execution"))?;
    let resolved = args.resolve_run_args(cfg)?;
    let (result, _memory) = run_prompt(&prompt, &resolved, SessionMemory::new()).await?;

    info!(steps = result.steps_taken, "run complete");
    println!("{}", result.answer);

    Ok(())
}
// grcov-excl-stop

// grcov-excl-start: exclude inline unit tests from production coverage
#[cfg(test)]
mod tests {
    use super::*;

    fn args(prompt: Option<&str>, model: Option<&str>, traces_dir: Option<&str>) -> PromptArgs {
        PromptArgs {
            prompt: prompt.map(str::to_string),
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
        let resolved = args(Some("hello"), Some("openai/gpt-4o"), None)
            .resolve_run_args(&cfg(Some("anthropic/claude-3-5-sonnet-20241022"), None))
            .unwrap();
        assert_eq!(resolved.model, "openai/gpt-4o");
    }

    #[test]
    fn config_model_used_when_cli_absent() {
        let resolved = args(Some("hello"), None, None)
            .resolve_run_args(&cfg(Some("anthropic/claude-3-5-sonnet-20241022"), None))
            .unwrap();
        assert_eq!(resolved.model, "anthropic/claude-3-5-sonnet-20241022");
    }

    #[test]
    fn missing_model_from_both_is_error() {
        let err = args(Some("hello"), None, None)
            .resolve_run_args(&cfg(None, None))
            .unwrap_err();
        assert!(err.to_string().contains("--model is required"));
    }

    #[test]
    fn cli_traces_dir_takes_precedence_over_config() {
        let resolved = args(Some("hello"), Some("openai/gpt-4o"), Some("/cli/traces"))
            .resolve_run_args(&cfg(None, Some("/cfg/traces")))
            .unwrap();
        assert_eq!(resolved.traces_dir, "/cli/traces");
    }

    #[test]
    fn config_traces_dir_used_when_cli_absent() {
        let resolved = args(Some("hello"), Some("openai/gpt-4o"), None)
            .resolve_run_args(&cfg(None, Some("/cfg/traces")))
            .unwrap();
        assert_eq!(resolved.traces_dir, "/cfg/traces");
    }

    #[test]
    fn default_traces_dir_when_both_absent() {
        let resolved = args(Some("hello"), Some("openai/gpt-4o"), None)
            .resolve_run_args(&cfg(None, None))
            .unwrap();
        assert_eq!(resolved.traces_dir, DEFAULT_TRACES_DIR);
    }

    #[test]
    fn tilde_in_traces_dir_is_expanded() {
        let resolved = args(Some("hello"), Some("openai/gpt-4o"), Some("~/my/traces"))
            .resolve_run_args(&cfg(None, None))
            .unwrap();
        assert!(!resolved.traces_dir.starts_with('~'));
        assert!(resolved.traces_dir.ends_with("/my/traces"));
    }

    #[test]
    fn prompt_is_optional_for_tui_dispatch() {
        assert_eq!(
            args(None, Some("openai/gpt-4o"), None)
                .prompt_text()
                .unwrap(),
            None
        );
    }

    #[test]
    fn prompt_is_passed_through_for_non_interactive_mode() {
        assert_eq!(
            args(Some("hello"), Some("openai/gpt-4o"), None)
                .prompt_text()
                .unwrap(),
            Some("hello".to_string())
        );
    }

    #[test]
    fn empty_prompt_is_rejected() {
        let err = args(Some("   "), Some("openai/gpt-4o"), None)
            .prompt_text()
            .unwrap_err();
        assert!(err.to_string().contains("prompt must not be empty"));
    }

    #[test]
    fn whitespace_only_model_is_error() {
        let err = args(Some("hello"), Some("   "), None)
            .resolve_run_args(&cfg(None, None))
            .unwrap_err();
        assert!(err.to_string().contains("must not be empty"));
    }
}
// grcov-excl-stop
