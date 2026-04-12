use std::io::{self, Stdout};

use anyhow::{Context, Result};
use crossterm::{
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, Terminal};

pub(crate) type AppTerminal = Terminal<CrosstermBackend<Stdout>>;

pub(crate) fn init_terminal() -> Result<AppTerminal> {
    let original_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), LeaveAlternateScreen, crossterm::cursor::Show);
        original_hook(info);
    }));

    enable_raw_mode().context("failed to enable raw mode")?;
    let mut stdout = io::stdout();

    if let Err(e) = execute!(stdout, EnterAlternateScreen) {
        let _ = disable_raw_mode();
        return Err(e).context("failed to enter alternate screen");
    }

    let backend = CrosstermBackend::new(stdout);
    Terminal::new(backend).map_err(|e| {
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), LeaveAlternateScreen);
        anyhow::anyhow!("failed to initialize terminal: {e}")
    })
}

pub(crate) fn restore_terminal(terminal: &mut AppTerminal) -> Result<()> {
    let r1 = disable_raw_mode().context("failed to disable raw mode");
    let r2 = execute!(terminal.backend_mut(), LeaveAlternateScreen)
        .context("failed to leave alternate screen");
    let r3 = terminal.show_cursor().context("failed to restore cursor");
    r1.and(r2).and(r3)
}
