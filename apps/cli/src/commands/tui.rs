use std::io::{self, Stdout};
use std::time::Duration;

use anyhow::{Context, Result};
use crossterm::{
    event::{self, Event, KeyCode, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Paragraph, Wrap},
    Frame, Terminal,
};
use tokio::sync::mpsc;
use tui_textarea::TextArea;

use crate::config::YaaiConfig;

use super::prompt::PromptArgs;
use super::runner::{run_prompt, PromptRunResult, ResolvedRunArgs};

type AppTerminal = Terminal<CrosstermBackend<Stdout>>;

#[derive(Debug, Clone, PartialEq, Eq)]
enum TranscriptRole {
    User,
    Assistant,
    Error,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TranscriptEntry {
    role: TranscriptRole,
    content: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum RunState {
    Idle,
    Running,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct AppState {
    transcript: Vec<TranscriptEntry>,
    status: String,
    run_state: RunState,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            transcript: Vec::new(),
            status: "Ready. Enter submits, Shift+Enter adds a newline, Ctrl+C exits.".to_string(),
            run_state: RunState::Idle,
        }
    }
}

impl AppState {
    fn can_submit(&self, input: &str) -> bool {
        self.run_state == RunState::Idle && !input.trim().is_empty()
    }

    fn start_run(&mut self, prompt: &str) {
        self.transcript.push(TranscriptEntry {
            role: TranscriptRole::User,
            content: prompt.to_string(),
        });
        self.status = "Running agent...".to_string();
        self.run_state = RunState::Running;
    }

    fn complete_run(&mut self, result: Result<PromptRunResult, String>) {
        match result {
            Ok(result) => {
                self.transcript.push(TranscriptEntry {
                    role: TranscriptRole::Assistant,
                    content: result.answer,
                });
                self.status = format!("Run complete in {} step(s).", result.steps_taken);
            }
            Err(err) => {
                self.transcript.push(TranscriptEntry {
                    role: TranscriptRole::Error,
                    content: err,
                });
                self.status = "Run failed.".to_string();
            }
        }
        self.run_state = RunState::Idle;
    }

    fn clear_status(&mut self) {
        if self.run_state == RunState::Idle {
            self.status =
                "Ready. Enter submits, Shift+Enter adds a newline, Ctrl+C exits.".to_string();
        }
    }

    fn transcript_text(&self) -> Text<'static> {
        if self.transcript.is_empty() {
            return Text::from(vec![Line::from(
                "No messages yet. Type a prompt below to start a run.",
            )]);
        }

        let mut lines = Vec::new();
        for entry in &self.transcript {
            let label = match entry.role {
                TranscriptRole::User => "You",
                TranscriptRole::Assistant => "Assistant",
                TranscriptRole::Error => "Error",
            };

            let mut first = true;
            for sub in entry.content.split('\n') {
                if first {
                    lines.push(Line::from(vec![
                        Span::styled(
                            format!("{label}: "),
                            Style::default().add_modifier(Modifier::BOLD),
                        ),
                        Span::raw(sub.to_string()),
                    ]));
                    first = false;
                } else {
                    lines.push(Line::from(Span::raw(sub.to_string())));
                }
            }
            lines.push(Line::from(""));
        }

        Text::from(lines)
    }
}

struct TuiApp {
    state: AppState,
    composer: TextArea<'static>,
    run_args: ResolvedRunArgs,
    result_rx: mpsc::UnboundedReceiver<Result<PromptRunResult, String>>,
    result_tx: mpsc::UnboundedSender<Result<PromptRunResult, String>>,
    active_task: Option<tokio::task::JoinHandle<()>>,
}

impl TuiApp {
    fn new(run_args: ResolvedRunArgs) -> Self {
        let (result_tx, result_rx) = mpsc::unbounded_channel();
        let mut composer = TextArea::default();
        composer.set_block(
            Block::default()
                .borders(Borders::ALL)
                .title("Prompt")
                .title_bottom("Enter submit | Shift+Enter newline"),
        );

        Self {
            state: AppState::default(),
            composer,
            run_args,
            result_rx,
            result_tx,
            active_task: None,
        }
    }

    async fn run(&mut self) -> Result<()> {
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

    fn footer_line(&self) -> Line<'static> {
        let state = match self.state.run_state {
            RunState::Idle => "idle",
            RunState::Running => "running",
        };

        Line::from(vec![
            Span::styled(
                format!("[{state}] "),
                Style::default().add_modifier(Modifier::BOLD),
            ),
            Span::raw(self.state.status.clone()),
        ])
    }
}

fn new_composer() -> TextArea<'static> {
    let mut composer = TextArea::default();
    composer.set_block(
        Block::default()
            .borders(Borders::ALL)
            .title("Prompt")
            .title_bottom("Enter submit | Shift+Enter newline"),
    );
    composer
}

fn composer_height(composer: &TextArea<'_>, terminal_width: u16) -> u16 {
    let wrap_width = (terminal_width as usize).max(24);
    let wrapped_hint = textwrap::wrap("Enter submit | Shift+Enter newline", wrap_width);
    let body_lines = composer.lines().len().max(1);
    let total = body_lines + wrapped_hint.len() + 2;
    total.min(8) as u16
}

fn init_terminal() -> Result<AppTerminal> {
    let original_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), LeaveAlternateScreen);
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

fn restore_terminal(terminal: &mut AppTerminal) -> Result<()> {
    disable_raw_mode().context("failed to disable raw mode")?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)
        .context("failed to leave alternate screen")?;
    terminal.show_cursor().context("failed to restore cursor")
}

pub async fn execute(args: &PromptArgs, cfg: &YaaiConfig) -> Result<()> {
    let run_args = args.resolve_run_args(cfg)?;
    let mut app = TuiApp::new(run_args);
    app.run().await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn submit_requires_non_empty_input_and_idle_state() {
        let state = AppState::default();
        assert!(!state.can_submit("   "));
        assert!(state.can_submit("hello"));
    }

    #[test]
    fn start_run_appends_user_message_and_sets_running() {
        let mut state = AppState::default();
        state.start_run("hello");

        assert_eq!(state.run_state, RunState::Running);
        assert_eq!(state.transcript.len(), 1);
        assert_eq!(state.transcript[0].role, TranscriptRole::User);
    }

    #[test]
    fn complete_run_appends_assistant_message_and_resets_idle() {
        let mut state = AppState::default();
        state.start_run("hello");
        state.complete_run(Ok(PromptRunResult {
            answer: "done".to_string(),
            steps_taken: 2,
        }));

        assert_eq!(state.run_state, RunState::Idle);
        assert_eq!(state.transcript.len(), 2);
        assert_eq!(state.transcript[1].role, TranscriptRole::Assistant);
        assert!(state.status.contains("2"));
    }

    #[test]
    fn complete_run_appends_error_message_and_resets_idle() {
        let mut state = AppState::default();
        state.start_run("hello");
        state.complete_run(Err("boom".to_string()));

        assert_eq!(state.run_state, RunState::Idle);
        assert_eq!(state.transcript.len(), 2);
        assert_eq!(state.transcript[1].role, TranscriptRole::Error);
        assert_eq!(state.status, "Run failed.");
    }

    #[test]
    fn escape_clears_ready_status() {
        let mut state = AppState {
            status: "custom".to_string(),
            ..AppState::default()
        };

        state.clear_status();

        assert!(state.status.contains("Ready."));
    }

    #[test]
    fn composer_height_is_bounded() {
        let mut composer = new_composer();
        composer.insert_str("one\ntwo\nthree\nfour\nfive\nsix\nseven");

        assert_eq!(composer_height(&composer, 80), 8);
    }

    #[test]
    fn transcript_text_is_placeholder_when_empty() {
        let state = AppState::default();
        let text = state.transcript_text();
        let content: String = text
            .lines
            .iter()
            .flat_map(|l| l.spans.iter())
            .map(|s| s.content.as_ref())
            .collect();
        assert!(content.contains("No messages"));
    }

    #[test]
    fn transcript_text_shows_all_roles() {
        let mut state = AppState::default();
        state.start_run("question");
        state.complete_run(Ok(PromptRunResult {
            answer: "answer".to_string(),
            steps_taken: 1,
        }));
        state.start_run("q2");
        state.complete_run(Err("boom".to_string()));

        let text = state.transcript_text();
        let content: String = text
            .lines
            .iter()
            .flat_map(|l| l.spans.iter())
            .map(|s| s.content.as_ref())
            .collect();
        assert!(content.contains("You"));
        assert!(content.contains("Assistant"));
        assert!(content.contains("Error"));
    }

    #[test]
    fn can_submit_returns_false_when_running() {
        let mut state = AppState::default();
        state.start_run("hello");
        assert!(!state.can_submit("more input"));
    }

    #[test]
    fn clear_status_noop_when_running() {
        let mut state = AppState::default();
        state.start_run("hello");
        let status_before = state.status.clone();
        state.clear_status();
        assert_eq!(state.status, status_before);
    }

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

    #[test]
    fn new_composer_has_one_empty_line() {
        let composer = new_composer();
        assert_eq!(composer.lines().len(), 1);
        assert_eq!(composer.lines()[0], "");
    }
}
