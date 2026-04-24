//! Short-term session memory for agent runs.
//!
//! [`SessionMemory`] holds the ordered list of messages/observations that
//! constitute the in-context history for one agent invocation.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The role of a message in the session history.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// The content of a memory entry — either plain text, a tool invocation made
/// by the assistant, or the result returned to the assistant after execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EntryContent {
    Text {
        text: String,
    },
    ToolCall {
        id: String,
        name: String,
        arguments: Value,
        /// Text emitted by the model before the tool call (e.g. chain-of-thought).
        /// Preserved so it can be replayed as part of the same assistant message.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reasoning: Option<String>,
    },
    ToolResult {
        tool_call_id: String,
        content: String,
    },
}

impl std::fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        })
    }
}

/// A single entry in session memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    pub role: Role,
    pub content: EntryContent,
    pub timestamp: DateTime<Utc>,
}

impl MemoryEntry {
    pub fn text(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: EntryContent::Text {
                text: content.into(),
            },
            timestamp: Utc::now(),
        }
    }
}

/// In-memory, session-scoped context store.
///
/// Entries accumulate in insertion order.
#[derive(Debug, Default, Clone)]
pub struct SessionMemory {
    entries: Vec<MemoryEntry>,
}

impl SessionMemory {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append an entry.
    pub fn push(&mut self, entry: MemoryEntry) {
        self.entries.push(entry);
    }

    /// Convenience: push a plain-text message.
    pub fn add(&mut self, role: Role, content: impl Into<String>) {
        self.push(MemoryEntry::text(role, content));
    }

    /// Convenience: push a structured entry.
    pub fn add_entry(&mut self, role: Role, content: EntryContent) {
        self.push(MemoryEntry {
            role,
            content,
            timestamp: chrono::Utc::now(),
        });
    }

    /// All entries in insertion order.
    pub fn entries(&self) -> &[MemoryEntry] {
        &self.entries
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_and_retrieve() {
        let mut mem = SessionMemory::new();
        mem.add(Role::User, "hello");
        mem.add(Role::Assistant, "hi there");

        assert_eq!(mem.len(), 2);
        assert!(
            matches!(&mem.entries()[0].content, EntryContent::Text { text } if text == "hello")
        );
        assert_eq!(mem.entries()[1].role, Role::Assistant);
    }

    #[test]
    fn push_directly_with_memory_entry() {
        let mut mem = SessionMemory::new();
        let entry = MemoryEntry::text(Role::Assistant, "direct push");
        mem.push(entry);

        assert_eq!(mem.len(), 1);
        assert_eq!(mem.entries()[0].role, Role::Assistant);
        assert!(
            matches!(&mem.entries()[0].content, EntryContent::Text { text } if text == "direct push")
        );
    }

    #[test]
    fn add_entry_stores_structured_content() {
        let mut mem = SessionMemory::new();
        mem.add_entry(
            Role::Assistant,
            EntryContent::ToolCall {
                id: "call_1".into(),
                name: "read".into(),
                arguments: serde_json::json!({ "file_path": "/LICENSE" }),
                reasoning: None,
            },
        );
        mem.add_entry(
            Role::User,
            EntryContent::ToolResult {
                tool_call_id: "call_1".into(),
                content: "MIT License".into(),
            },
        );

        assert_eq!(mem.len(), 2);
        assert!(matches!(
            &mem.entries()[0].content,
            EntryContent::ToolCall { id, .. } if id == "call_1"
        ));
        assert!(matches!(
            &mem.entries()[1].content,
            EntryContent::ToolResult { tool_call_id, .. } if tool_call_id == "call_1"
        ));
    }

    #[test]
    fn is_empty_on_new() {
        assert!(SessionMemory::new().is_empty());
    }

    #[test]
    fn role_serde_round_trip() {
        for (role, expected) in [
            (Role::System, "system"),
            (Role::User, "user"),
            (Role::Assistant, "assistant"),
            (Role::Tool, "tool"),
        ] {
            let json = serde_json::to_string(&role).unwrap();
            assert_eq!(json, format!("\"{}\"", expected));
            let r2: Role = serde_json::from_str(&json).unwrap();
            assert_eq!(r2, role);
        }

        let entry = MemoryEntry::text(Role::User, "hello");
        let json = serde_json::to_string(&entry).unwrap();
        let e2: MemoryEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(e2.role, Role::User);
        assert!(matches!(e2.content, EntryContent::Text { text } if text == "hello"));
    }
}
