use std::time::Duration;

use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyModifiers};
use ratatui::{
    layout::{Constraint, Direction, Layout},
    style::{Modifier, Style},
    text::Line,
    widgets::{Block, Borders, Paragraph, Wrap},
    Frame,
};
use tokio::sync::mpsc;
use tui_textarea::TextArea;

use crate::commands::runner::{run_prompt, PromptRunResult, ResolvedRunArgs};

use super::composer::{composer_height, new_composer};
use super::state::{AppState, RunState};
use super::terminal::{init_terminal, restore_terminal, AppTerminal};

pub(crate) struct TuiApp {
    pub(crate) state: AppState,
    composer: TextArea<'static>,
    run_args: ResolvedRunArgs,
    result_rx: mpsc::UnboundedReceiver<Result<PromptRunResult, String>>,
    result_tx: mpsc::UnboundedSender<Result<PromptRunResult, String>>,
    active_task: Option<tokio::task::JoinHandle<()>>,
}

impl TuiApp {
    pub(crate) fn new(run_args: ResolvedRunArgs) -> Self {
        let (result_tx, result_rx) = mpsc::unbounded_channel();

        Self {
            state: AppState::default(),
            composer: new_composer(),
            run_args,
            result_rx,
            result_tx,
            active_task: None,
        }
    }

    pub(crate) async fn run(&mut self) -> Result<()> {
        let mut terminal = init_terminal()?;

        let run_result = self.event_loop(&mut terminal).await;
        let restore_result = restore_terminal(&mut terminal);
        run_result.and(restore_result)
    }

    async fn event_loop(&mut self, terminal: &mut AppTerminal) -> Result<()> {
        let mut dirty = true;
        loop {
            while let Ok(result) = self.result_rx.try_recv() {
                self.state.complete_run(result);
                dirty = true;
            }

            if dirty {
                terminal.draw(|frame| self.render(frame))?;
                dirty = false;
            }

            if event::poll(Duration::from_millis(50))? {
                let evt = event::read()?;
                if self.handle_event(evt).await? {
                    return Ok(());
                }
                dirty = true;
            }
        }
    }

    async fn handle_event(&mut self, evt: Event) -> Result<bool> {
        if let Event::Key(key) = evt {
            if key.kind != event::KeyEventKind::Press {
                return Ok(false);
            }

            if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
                if let Some(task) = self.active_task.take() {
                    task.abort();
                }
                return Ok(true);
            }

            match key.code {
                KeyCode::Esc => self.state.clear_status(),
                KeyCode::Enter if !key.modifiers.contains(KeyModifiers::SHIFT) => {
                    self.submit_current_prompt();
                }
                _ => {
                    self.composer.input(key);
                }
            }
        }

        Ok(false)
    }

    fn submit_current_prompt(&mut self) {
        let input = self.composer.lines().join("\n");
        if !self.state.can_submit(&input) {
            return;
        }

        self.state.start_run(&input);
        self.composer = new_composer();
        let tx = self.result_tx.clone();
        let run_args = self.run_args.clone();

        if let Some(old) = self.active_task.take() {
            old.abort();
        }
        let handle = tokio::spawn(async move {
            let result = run_prompt(&input, &run_args)
                .await
                .map_err(|err| err.to_string());
            let _ = tx.send(result);
        });
        self.active_task = Some(handle);
    }

    fn render(&mut self, frame: &mut Frame<'_>) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(1),
                Constraint::Length(composer_height(&self.composer, frame.area().width)),
                Constraint::Length(1),
            ])
            .split(frame.area());

        let transcript = Paragraph::new(self.state.transcript_text())
            .block(Block::default().borders(Borders::ALL).title("Transcript"))
            .wrap(Wrap { trim: false });
        frame.render_widget(transcript, chunks[0]);

        frame.render_widget(&self.composer, chunks[1]);

        let footer = Paragraph::new(self.footer_line());
        frame.render_widget(footer, chunks[2]);
    }

    pub(crate) fn footer_line(&self) -> Line<'static> {
        let state = match self.state.run_state {
            RunState::Idle => "idle",
            RunState::Running => "running",
        };

        Line::from(vec![
            ratatui::text::Span::styled(
                format!("[{state}] "),
                Style::default().add_modifier(Modifier::BOLD),
            ),
            ratatui::text::Span::raw(self.state.status.clone()),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn footer_line_shows_idle_state() {
        let run_args = ResolvedRunArgs {
            model: "openai/gpt-4o".to_string(),
            traces_dir: "traces".to_string(),
        };
        let app = TuiApp::new(run_args);
        let line = app.footer_line();
        let content: String = line.spans.iter().map(|s| s.content.as_ref()).collect();
        assert!(content.contains("idle"));
    }

    #[test]
    fn footer_line_shows_running_state() {
        let run_args = ResolvedRunArgs {
            model: "openai/gpt-4o".to_string(),
            traces_dir: "traces".to_string(),
        };
        let mut app = TuiApp::new(run_args);
        app.state.run_state = RunState::Running;
        let line = app.footer_line();
        let content: String = line.spans.iter().map(|s| s.content.as_ref()).collect();
        assert!(content.contains("running"));
    }
}
