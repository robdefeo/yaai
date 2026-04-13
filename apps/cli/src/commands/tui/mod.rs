mod app;
mod composer;
mod state;
mod terminal;

use anyhow::Result;

use crate::config::YaaiConfig;

use super::prompt::PromptArgs;
use app::TuiApp;

pub async fn execute(args: &PromptArgs, cfg: &YaaiConfig) -> Result<()> {
    let run_args = args.resolve_run_args(cfg)?;
    let mut app = TuiApp::new(run_args);
    app.run().await
}
