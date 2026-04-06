//! yaai — POC Agent Harness CLI

mod commands;

use anyhow::Result;
use clap::Parser;
use commands::prompt::PromptArgs;
use yaai_tracer::init_tracing;

#[derive(Parser)]
#[command(name = "yaai", about = "POC Agent Harness", version)]
struct Cli {
    #[arg(
        long,
        global = true,
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
    let _ = init_tracing(cli.json_logs);
    commands::prompt::execute(cli.args).await
    // grcov-excl-stop
}
