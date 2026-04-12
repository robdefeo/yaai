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

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AppState {
    pub(crate) transcript: Vec<TranscriptEntry>,
    pub(crate) status: String,
    pub(crate) run_state: RunState,
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
    }

    pub(crate) fn complete_run(&mut self, result: Result<PromptRunResult, String>) {
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

    pub(crate) fn clear_status(&mut self) {
        if self.run_state == RunState::Idle {
            self.status =
                "Ready. Enter submits, Shift+Enter adds a newline, Ctrl+C exits.".to_string();
        }
    }

    pub(crate) fn transcript_text(&self) -> Text<'static> {
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
