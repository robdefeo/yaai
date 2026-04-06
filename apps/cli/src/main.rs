//! yaai — POC Agent Harness CLI

mod commands;

use anyhow::Result;
use clap::{ArgAction, Parser};
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
        help = "Emit logs as structured JSON instead of pretty-printed text. \
                Useful when piping output to a log aggregator or structured logging pipeline."
    )]
    json_logs: bool,

    #[command(flatten)]
    args: PromptArgs,
}

#[tokio::main]
async fn main() -> Result<()> {
    // grcov-excl-start
    let cli = Cli::parse();
    init_tracing(cli.json_logs)?;
    commands::prompt::execute(cli.args).await
    // grcov-excl-stop
}
