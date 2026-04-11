//! yaai — POC Agent Harness CLI

mod commands;
mod config;

use anyhow::Result;
use clap::{ArgAction, CommandFactory, FromArgMatches, Parser};
use commands::prompt::PromptArgs;
use yaai_tracer::init_tracing;

#[derive(Parser)]
#[command(
    name = "yaai",
    about = "POC Agent Harness",
    version,
    max_term_width = 100,
    disable_help_flag = true,
    disable_version_flag = true,
    help_template = "\
{before-help}{about-with-newline}\
{usage-heading} {usage}\n\n\
{all-args}{after-help}\
"
)]
struct Cli {
    #[arg(
        short = 'h',
        long = "help",
        action = ArgAction::Help,
        global = true,
        display_order = 10,
        help_heading = "Options",
        help = "Print help"
    )]
    _help: Option<bool>,

    #[arg(
        short = 'V',
        long = "version",
        action = ArgAction::Version,
        global = true,
        display_order = 11,
        help_heading = "Options",
        help = "Print version"
    )]
    _version: Option<bool>,

    #[arg(
        long,
        global = true,
        display_order = 12,
        help_heading = "Options",
        help = "Write logs as structured JSON instead of human-readable text. \
                Useful when feeding the log file into a log aggregator or structured logging pipeline."
    )]
    json_logs: bool,

    #[command(flatten)]
    args: PromptArgs,
}

#[tokio::main]
async fn main() -> Result<()> {
    // grcov-excl-start
    let config_path = config::config_path_display();

    let matches = Cli::command()
        .mut_arg("model", |a| {
            a.help(format!(
                "The model to use, specified as provider/model (e.g. openai/gpt-4o, \
                 anthropic/claude-3-5-sonnet-20241022). The corresponding API key must be \
                 set in the environment (OPENAI_API_KEY or ANTHROPIC_API_KEY). \
                 Falls back to `model` in {config_path} if not set."
            ))
        })
        .mut_arg("traces_dir", |a| {
            a.help(format!(
                "Directory where trace files are written after each run. \
                 Each run produces a file named <run-id>.ndjson containing \
                 newline-delimited JSON (NDJSON) — one event object per line. \
                 Falls back to `traces_dir` in {config_path}, then \"traces\"."
            ))
        })
        .mut_arg("json_logs", |a| {
            a.help(format!(
                "Write logs as structured JSON instead of human-readable text. \
                 Logs are written to <data_dir>/yaai/logs/yaai.YYYY-MM-DD.log \
                 (e.g. ~/Library/Application Support/yaai/logs/ on macOS). \
                 Falls back to `json_logs` in {config_path}."
            ))
        })
        .get_matches();
    let cli = Cli::from_arg_matches(&matches)?;
    let cfg = config::load()?;

    let json_logs = cli.json_logs || cfg.json_logs.unwrap_or(false);
    let log_dir = dirs::data_local_dir()
        .unwrap_or_else(std::env::temp_dir)
        .join("yaai")
        .join("logs");
    let _log_guard = init_tracing(json_logs, &log_dir);

    commands::prompt::execute(cli.args, &cfg).await
    // grcov-excl-stop
}
