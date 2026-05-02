use ratatui::{
    style::{Modifier, Style},
    text::{Line, Span, Text},
};

use crate::commands::runner::PromptRunResult;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum TranscriptRole {
    User,
    Assistant,
    Error,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TranscriptEntry {
    pub(crate) role: TranscriptRole,
    pub(crate) content: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum RunState {
    Idle,
    Running,
}

/// Newline-gated streaming buffer — mirrors the `MarkdownStreamCollector` pattern from codex.
///
/// Raw token deltas accumulate in `raw`. `committed_end` tracks the byte offset after the last
/// `\n` in `raw`: everything before that offset is "committed" (safe to render as complete lines).
/// The slice from `committed_end` to the end is the partial current line shown with a cursor.
/// On finalisation the pending slice is flushed even without a trailing newline.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct StreamingBuffer {
    raw: String,
    committed_end: usize,
}

impl StreamingBuffer {
    fn push(&mut self, token: &str) {
        self.raw.push_str(token);
        // Advance committed_end to after the last newline in the buffer.
        if token.contains('\n') {
            if let Some(idx) = self.raw.rfind('\n') {
                self.committed_end = idx + 1;
            }
        }
    }

    fn committed(&self) -> &str {
        &self.raw[..self.committed_end]
    }

    fn pending(&self) -> &str {
        &self.raw[self.committed_end..]
    }

    fn clear(&mut self) {
        self.raw.clear();
        self.committed_end = 0;
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AppState {
    pub(crate) transcript: Vec<TranscriptEntry>,
    pub(crate) status: String,
    pub(crate) run_state: RunState,
    streaming: StreamingBuffer,
    streaming_active: bool,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            transcript: Vec::new(),
            status: "Ready. Enter submits, Shift+Enter adds a newline, PageUp/PageDown scrolls, Ctrl+C exits.".to_string(),
            run_state: RunState::Idle,
            streaming: StreamingBuffer::default(),
            streaming_active: false,
        }
    }
}

impl AppState {
    pub(crate) fn can_submit(&self, input: &str) -> bool {
        self.run_state == RunState::Idle && !input.trim().is_empty()
    }

    pub(crate) fn start_run(&mut self, prompt: &str) {
        self.transcript.push(TranscriptEntry {
            role: TranscriptRole::User,
            content: prompt.to_string(),
        });
        self.status = "Running agent...".to_string();
        self.run_state = RunState::Running;
        self.streaming.clear();
        self.streaming_active = true;
    }

    /// Push a raw token delta into the newline-gated buffer.
    ///
    /// Tokens accumulate in `streaming.raw`. `committed_end` advances only when a `\n`
    /// arrives, so partial lines are held back and never rendered mid-character — matching
    /// the codex `MarkdownStreamCollector` contract.  The partial current line is always
    /// shown with a `▊` cursor so the user sees progress within a line too.
    pub(crate) fn append_token(&mut self, token: String) {
        self.streaming.push(&token);
    }

    pub(crate) fn complete_run(&mut self, result: Result<PromptRunResult, String>) {
        // Drain the streaming buffer before clearing it so we can commit the
        // full accumulated text (all intermediate reasoning + final answer) as
        // a permanent transcript entry instead of only the last answer.
        let accumulated = std::mem::take(&mut self.streaming).raw;
        self.streaming_active = false;
        match result {
            Ok(result) => {
                // Use accumulated streaming text when available — it contains all
                // intermediate step reasoning plus the final answer.  Fall back to
                // result.answer for non-streaming runs or when nothing was streamed.
                let content = if accumulated.is_empty() {
                    result.answer
                } else {
                    accumulated
                };
                self.transcript.push(TranscriptEntry {
                    role: TranscriptRole::Assistant,
                    content,
                });
                self.status = format!("Run complete in {} step(s).", result.steps_taken);
            }
            Err(err) => {
                if !accumulated.is_empty() {
                    self.transcript.push(TranscriptEntry {
                        role: TranscriptRole::Assistant,
                        content: accumulated,
                    });
                }
                self.transcript.push(TranscriptEntry {
                    role: TranscriptRole::Error,
                    content: err,
                });
                self.status = "Run failed.".to_string();
            }
        }
        self.run_state = RunState::Idle;
    }

    pub(crate) fn clear_status(&mut self) {
        if self.run_state == RunState::Idle {
            self.status =
                "Ready. Enter submits, Shift+Enter adds a newline, PageUp/PageDown scrolls, Ctrl+C exits.".to_string();
        }
    }

    pub(crate) fn transcript_text(&self) -> Text<'static> {
        if self.transcript.is_empty() && !self.streaming_active {
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

        // Streaming live tail — newline-gated, matching codex MarkdownStreamCollector semantics:
        //
        // Committed source (everything up to the last \n) is rendered as stable complete lines.
        // The pending slice (partial current line) is rendered separately with a block cursor so
        // the user sees in-line progress without exposing mid-line content as a completed row.
        if self.streaming_active {
            let committed = self.streaming.committed();
            let pending = self.streaming.pending();

            // Render committed lines (newline-terminated, stable).
            let mut first_streaming_line = true;
            for sub in committed.split('\n') {
                // split('\n') on "a\nb\n" → ["a", "b", ""] — skip all empty segments,
                // including the sole "" produced by splitting an empty committed string
                // before any newline has arrived.
                if sub.is_empty() {
                    continue;
                }
                if first_streaming_line {
                    lines.push(Line::from(vec![
                        Span::styled(
                            "Assistant: ".to_string(),
                            Style::default().add_modifier(Modifier::BOLD),
                        ),
                        Span::raw(sub.to_string()),
                    ]));
                    first_streaming_line = false;
                } else {
                    lines.push(Line::from(Span::raw(sub.to_string())));
                }
            }

            // Render pending partial line with cursor.
            let cursor_line = format!("{}▊", pending);
            if first_streaming_line {
                // Nothing committed yet — pending is the first visible line.
                lines.push(Line::from(vec![
                    Span::styled(
                        "Assistant: ".to_string(),
                        Style::default().add_modifier(Modifier::BOLD),
                    ),
                    Span::raw(cursor_line),
                ]));
            } else {
                lines.push(Line::from(Span::raw(cursor_line)));
            }
        }

        Text::from(lines)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::runner::PromptRunResult;

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
    fn complete_run_error_with_streamed_content_emits_both_entries() {
        let mut state = AppState::default();
        state.start_run("hello");
        state.append_token("partial answer".to_string());
        state.complete_run(Err("boom".to_string()));

        assert_eq!(state.transcript.len(), 3); // user + assistant + error
        assert_eq!(state.transcript[1].role, TranscriptRole::Assistant);
        assert_eq!(state.transcript[1].content, "partial answer");
        assert_eq!(state.transcript[2].role, TranscriptRole::Error);
        assert_eq!(state.transcript[2].content, "boom");
        assert_eq!(state.run_state, RunState::Idle);
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
}
