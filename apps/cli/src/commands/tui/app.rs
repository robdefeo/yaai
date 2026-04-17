use std::time::Duration;

use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyModifiers};
use ratatui::{
    layout::{Constraint, Direction, Layout},
    style::{Modifier, Style},
    text::{Line, Text},
    widgets::{Block, Borders, Paragraph, Wrap},
    Frame,
};
use tokio::sync::mpsc;
use tui_textarea::TextArea;
use yaai_memory::SessionMemory;

use crate::commands::runner::{run_prompt, PromptRunResult, ResolvedRunArgs};

use super::composer::{composer_height, new_composer};
use super::state::{AppState, RunState};
use super::terminal::{init_terminal, restore_terminal, AppTerminal};

type RunResult = Result<(PromptRunResult, SessionMemory), String>;

pub(crate) struct TuiApp {
    pub(crate) state: AppState,
    composer: TextArea<'static>,
    run_args: ResolvedRunArgs,
    pub(crate) session_memory: SessionMemory,
    result_rx: mpsc::UnboundedReceiver<RunResult>,
    result_tx: mpsc::UnboundedSender<RunResult>,
    active_task: Option<tokio::task::JoinHandle<()>>,
    /// `usize::MAX` means "pinned to bottom — auto-follow new content".
    scroll_offset: usize,
    last_max_scroll: usize,
    last_viewport_height: usize,
}

impl TuiApp {
    pub(crate) fn new(run_args: ResolvedRunArgs) -> Self {
        let (result_tx, result_rx) = mpsc::unbounded_channel();

        Self {
            state: AppState::default(),
            composer: new_composer(),
            run_args,
            session_memory: SessionMemory::new(),
            result_rx,
            result_tx,
            active_task: None,
            scroll_offset: usize::MAX,
            last_max_scroll: 0,
            last_viewport_height: 0,
        }
    }

    pub(crate) async fn run(&mut self) -> Result<()> {
        let mut terminal = init_terminal()?;

        let run_result = self.event_loop(&mut terminal).await;
        let restore_result = restore_terminal(&mut terminal);
        run_result.and(restore_result)
    }

    fn process_run_result(&mut self, result: RunResult) {
        match result {
            Ok((run_result, updated_memory)) => {
                self.session_memory = updated_memory;
                self.state.complete_run(Ok(run_result));
            }
            Err(e) => {
                self.state.complete_run(Err(e));
            }
        }
    }

    async fn event_loop(&mut self, terminal: &mut AppTerminal) -> Result<()> {
        let mut dirty = true;
        loop {
            while let Ok(result) = self.result_rx.try_recv() {
                self.process_run_result(result);
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
                KeyCode::PageUp => {
                    let step = self.last_viewport_height.max(1);
                    let current = self.scroll_offset.min(self.last_max_scroll);
                    self.scroll_offset = current.saturating_sub(step);
                }
                KeyCode::PageDown => {
                    let step = self.last_viewport_height.max(1);
                    let current = self.scroll_offset.min(self.last_max_scroll);
                    let next = current.saturating_add(step);
                    self.scroll_offset = if next >= self.last_max_scroll {
                        usize::MAX
                    } else {
                        next
                    };
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
        self.scroll_offset = usize::MAX;
        self.composer = new_composer();
        let tx = self.result_tx.clone();
        let run_args = self.run_args.clone();
        let memory_snapshot = self.session_memory.clone();

        if let Some(old) = self.active_task.take() {
            old.abort();
        }
        while self.result_rx.try_recv().is_ok() {}
        let handle = tokio::spawn(async move {
            let result = run_prompt(&input, &run_args, memory_snapshot)
                .await
                .map_err(|e| e.to_string());
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

        let transcript_area = chunks[0];
        let inner_width = transcript_area.width.saturating_sub(2);
        let viewport_height = transcript_area.height.saturating_sub(2) as usize;

        let text = self.state.transcript_text();
        let total_lines = count_wrapped_lines(&text, inner_width);
        let max_scroll = total_lines.saturating_sub(viewport_height);

        let display_offset = self.scroll_offset.min(max_scroll) as u16;
        self.last_max_scroll = max_scroll;
        self.last_viewport_height = viewport_height;

        let transcript = Paragraph::new(text)
            .block(Block::default().borders(Borders::ALL).title("Transcript"))
            .wrap(Wrap { trim: false })
            .scroll((display_offset, 0));
        frame.render_widget(transcript, transcript_area);

        frame.render_widget(&self.composer, chunks[1]);

        let footer = Paragraph::new(self.footer_line());
        frame.render_widget(footer, chunks[2]);
    }
}

impl Drop for TuiApp {
    fn drop(&mut self) {
        if let Some(task) = self.active_task.take() {
            task.abort();
        }
    }
}

impl TuiApp {
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

/// Counts the total number of display lines that `text` will occupy when
/// rendered at `width` columns with word-wrap enabled (matching the
/// `Wrap { trim: false }` setting used on the transcript `Paragraph`).
fn count_wrapped_lines(text: &Text<'_>, width: u16) -> usize {
    let w = width.max(1) as usize;
    text.lines
        .iter()
        .map(|line| {
            let raw: String = line.spans.iter().map(|s| s.content.as_ref()).collect();
            if raw.is_empty() {
                1
            } else {
                textwrap::wrap(&raw, w).len()
            }
        })
        .sum::<usize>()
        .max(1)
}

// grcov-excl-start: exclude inline unit tests from production coverage
#[cfg(test)]
mod tests {
    use crossterm::event::{KeyEvent, KeyEventKind, KeyEventState};
    use yaai_memory::SessionMemory;

    use crate::commands::runner::PromptRunResult;

    use super::*;

    fn make_app() -> TuiApp {
        TuiApp::new(ResolvedRunArgs {
            model: "openai/gpt-4o".to_string(),
            traces_dir: "traces".to_string(),
        })
    }

    fn key_press(code: KeyCode, modifiers: KeyModifiers) -> Event {
        Event::Key(KeyEvent {
            code,
            modifiers,
            kind: KeyEventKind::Press,
            state: KeyEventState::NONE,
        })
    }

    fn key_release(code: KeyCode) -> Event {
        Event::Key(KeyEvent {
            code,
            modifiers: KeyModifiers::NONE,
            kind: KeyEventKind::Release,
            state: KeyEventState::NONE,
        })
    }

    // --- TuiApp::new ---

    #[test]
    fn new_starts_with_idle_run_state() {
        let app = make_app();
        assert_eq!(app.state.run_state, RunState::Idle);
    }

    #[test]
    fn new_starts_with_empty_composer() {
        let app = make_app();
        assert_eq!(app.composer.lines().len(), 1);
        assert_eq!(app.composer.lines()[0], "");
    }

    #[test]
    fn new_starts_with_no_active_task() {
        let app = make_app();
        assert!(app.active_task.is_none());
    }

    // --- handle_event ---

    #[tokio::test]
    async fn handle_event_key_release_is_ignored() {
        let mut app = make_app();
        let exiting = app.handle_event(key_release(KeyCode::Enter)).await.unwrap();
        assert!(!exiting);
    }

    #[tokio::test]
    async fn handle_event_non_key_event_returns_false() {
        let mut app = make_app();
        let exiting = app.handle_event(Event::FocusGained).await.unwrap();
        assert!(!exiting);
    }

    #[tokio::test]
    async fn handle_event_ctrl_c_returns_true() {
        let mut app = make_app();
        let exiting = app
            .handle_event(key_press(KeyCode::Char('c'), KeyModifiers::CONTROL))
            .await
            .unwrap();
        assert!(exiting);
    }

    #[tokio::test]
    async fn handle_event_ctrl_c_aborts_active_task() {
        let mut app = make_app();
        let handle = tokio::spawn(async { std::future::pending::<()>().await });
        app.active_task = Some(handle);

        app.handle_event(key_press(KeyCode::Char('c'), KeyModifiers::CONTROL))
            .await
            .unwrap();

        assert!(app.active_task.is_none());
    }

    #[tokio::test]
    async fn handle_event_esc_clears_status() {
        let mut app = make_app();
        app.state.status = "custom status".to_string();

        let exiting = app
            .handle_event(key_press(KeyCode::Esc, KeyModifiers::NONE))
            .await
            .unwrap();

        assert!(!exiting);
        assert!(app.state.status.contains("Ready."));
    }

    #[tokio::test]
    async fn handle_event_regular_key_appends_to_composer() {
        let mut app = make_app();

        app.handle_event(key_press(KeyCode::Char('h'), KeyModifiers::NONE))
            .await
            .unwrap();
        app.handle_event(key_press(KeyCode::Char('i'), KeyModifiers::NONE))
            .await
            .unwrap();

        assert_eq!(app.composer.lines()[0], "hi");
    }

    #[tokio::test]
    async fn handle_event_enter_submits_when_idle_with_text() {
        let mut app = make_app();
        app.handle_event(key_press(KeyCode::Char('x'), KeyModifiers::NONE))
            .await
            .unwrap();
        app.handle_event(key_press(KeyCode::Enter, KeyModifiers::NONE))
            .await
            .unwrap();

        assert_eq!(app.state.run_state, RunState::Running);
        assert_eq!(app.state.transcript.len(), 1);
        assert!(app.active_task.is_some());
        assert_eq!(app.composer.lines()[0], "");
    }

    #[tokio::test]
    async fn handle_event_shift_enter_does_not_submit() {
        let mut app = make_app();
        app.handle_event(key_press(KeyCode::Char('x'), KeyModifiers::NONE))
            .await
            .unwrap();
        app.handle_event(key_press(KeyCode::Enter, KeyModifiers::SHIFT))
            .await
            .unwrap();

        assert_eq!(app.state.run_state, RunState::Idle);
        assert!(app.active_task.is_none());
    }

    // --- submit_current_prompt ---

    #[tokio::test]
    async fn submit_current_prompt_noop_on_empty_input() {
        let mut app = make_app();
        app.submit_current_prompt();
        assert_eq!(app.state.run_state, RunState::Idle);
        assert!(app.active_task.is_none());
    }

    #[tokio::test]
    async fn submit_current_prompt_noop_when_already_running() {
        let mut app = make_app();
        app.state.run_state = RunState::Running;
        app.composer.insert_str("some text");
        app.submit_current_prompt();
        assert_eq!(app.state.transcript.len(), 0);
        assert!(app.active_task.is_none());
    }

    #[tokio::test]
    async fn submit_current_prompt_clears_composer_and_spawns_task() {
        let mut app = make_app();
        app.composer.insert_str("hello");
        app.submit_current_prompt();

        assert_eq!(app.state.run_state, RunState::Running);
        assert_eq!(app.state.transcript.len(), 1);
        assert_eq!(app.state.transcript[0].content, "hello");
        assert_eq!(app.composer.lines()[0], "");
        assert!(app.active_task.is_some());
    }

    #[tokio::test]
    async fn submit_current_prompt_aborts_previous_task() {
        let mut app = make_app();
        let first_handle = tokio::spawn(async { std::future::pending::<()>().await });
        app.active_task = Some(first_handle);

        app.composer.insert_str("new prompt");
        app.submit_current_prompt();

        assert!(app.active_task.is_some());
    }

    // --- result channel ---

    #[tokio::test]
    async fn result_channel_ok_completes_run() {
        let mut app = make_app();
        app.state.start_run("test");

        app.result_tx
            .send(Ok((
                PromptRunResult {
                    answer: "done".to_string(),
                    steps_taken: 3,
                },
                SessionMemory::new(),
            )))
            .unwrap();

        while let Ok(result) = app.result_rx.try_recv() {
            app.process_run_result(result);
        }

        assert_eq!(app.state.run_state, RunState::Idle);
        assert!(app.state.status.contains("3"));
    }

    #[tokio::test]
    async fn result_channel_err_records_error_entry() {
        let mut app = make_app();
        app.state.start_run("test");

        app.result_tx.send(Err("boom".to_string())).unwrap();

        while let Ok(result) = app.result_rx.try_recv() {
            app.process_run_result(result);
        }

        assert_eq!(app.state.run_state, RunState::Idle);
        assert_eq!(app.state.status, "Run failed.");
    }

    // --- footer_line ---

    #[test]
    fn footer_line_shows_idle_state() {
        let app = make_app();
        let line = app.footer_line();
        let content: String = line.spans.iter().map(|s| s.content.as_ref()).collect();
        assert!(content.contains("idle"));
    }

    #[test]
    fn footer_line_shows_running_state() {
        let mut app = make_app();
        app.state.run_state = RunState::Running;
        let line = app.footer_line();
        let content: String = line.spans.iter().map(|s| s.content.as_ref()).collect();
        assert!(content.contains("running"));
    }

    // --- scrolling ---

    #[test]
    fn new_starts_pinned_to_bottom() {
        let app = make_app();
        assert_eq!(app.scroll_offset, usize::MAX);
    }

    #[tokio::test]
    async fn page_up_from_bottom_scrolls_up_by_half_viewport() {
        let mut app = make_app();
        app.last_max_scroll = 20;
        app.last_viewport_height = 10;

        app.handle_event(key_press(KeyCode::PageUp, KeyModifiers::NONE))
            .await
            .unwrap();

        // usize::MAX.min(20) = 20, step = 10, result = 10
        assert_eq!(app.scroll_offset, 10);
    }

    #[tokio::test]
    async fn page_up_from_position_scrolls_up() {
        let mut app = make_app();
        app.scroll_offset = 10;
        app.last_max_scroll = 20;
        app.last_viewport_height = 10;

        app.handle_event(key_press(KeyCode::PageUp, KeyModifiers::NONE))
            .await
            .unwrap();

        assert_eq!(app.scroll_offset, 0);
    }

    #[tokio::test]
    async fn page_up_clamps_to_zero() {
        let mut app = make_app();
        app.scroll_offset = 2;
        app.last_max_scroll = 20;
        app.last_viewport_height = 10;

        app.handle_event(key_press(KeyCode::PageUp, KeyModifiers::NONE))
            .await
            .unwrap();

        assert_eq!(app.scroll_offset, 0);
    }

    #[tokio::test]
    async fn page_down_re_pins_to_bottom_when_reaching_end() {
        let mut app = make_app();
        app.scroll_offset = 15;
        app.last_max_scroll = 20;
        app.last_viewport_height = 10;

        app.handle_event(key_press(KeyCode::PageDown, KeyModifiers::NONE))
            .await
            .unwrap();

        // 15 + 10 = 25 >= last_max_scroll(20) → pinned
        assert_eq!(app.scroll_offset, usize::MAX);
    }

    #[tokio::test]
    async fn page_down_from_position_scrolls_down() {
        let mut app = make_app();
        app.scroll_offset = 5;
        app.last_max_scroll = 20;
        app.last_viewport_height = 10;

        app.handle_event(key_press(KeyCode::PageDown, KeyModifiers::NONE))
            .await
            .unwrap();

        assert_eq!(app.scroll_offset, 15);
    }

    #[tokio::test]
    async fn page_down_when_already_pinned_stays_pinned() {
        let mut app = make_app();
        // scroll_offset = usize::MAX, last_max_scroll = 20 → current = 20
        app.last_max_scroll = 20;
        app.last_viewport_height = 10;

        app.handle_event(key_press(KeyCode::PageDown, KeyModifiers::NONE))
            .await
            .unwrap();

        assert_eq!(app.scroll_offset, usize::MAX);
    }

    #[tokio::test]
    async fn submit_resets_scroll_to_bottom() {
        let mut app = make_app();
        app.scroll_offset = 5;

        app.handle_event(key_press(KeyCode::Char('x'), KeyModifiers::NONE))
            .await
            .unwrap();
        app.handle_event(key_press(KeyCode::Enter, KeyModifiers::NONE))
            .await
            .unwrap();

        assert_eq!(app.scroll_offset, usize::MAX);
    }

    // --- count_wrapped_lines ---

    #[test]
    fn count_wrapped_lines_single_short_line() {
        let text = Text::from("hello");
        assert_eq!(count_wrapped_lines(&text, 80), 1);
    }

    #[test]
    fn count_wrapped_lines_empty_line_counts_as_one() {
        let text = Text::from("");
        assert_eq!(count_wrapped_lines(&text, 80), 1);
    }

    #[test]
    fn count_wrapped_lines_wraps_long_line() {
        // 20 chars wide, a 40-char string → 2 wrapped lines
        let text = Text::from("a".repeat(40));
        assert_eq!(count_wrapped_lines(&text, 20), 2);
    }

    #[test]
    fn count_wrapped_lines_sums_multiple_lines() {
        let text = Text::from(vec![
            ratatui::text::Line::from("short"),
            ratatui::text::Line::from("a".repeat(40)),
        ]);
        // 1 + 2 = 3 at width 20
        assert_eq!(count_wrapped_lines(&text, 20), 3);
    }
}
// grcov-excl-stop
