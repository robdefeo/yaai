//! Short-term session memory for agent runs.
//!
//! [`SessionMemory`] holds the ordered list of messages/observations that
//! constitute the in-context history for one agent invocation.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// The role of a message in the session history.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
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
    pub content: String,
    pub timestamp: DateTime<Utc>,
}

impl MemoryEntry {
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
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

    /// Convenience: push a message by role + content.
    pub fn add(&mut self, role: Role, content: impl Into<String>) {
        self.push(MemoryEntry::new(role, content));
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

    /// Remove all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
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
        assert_eq!(mem.entries()[0].content, "hello");
        assert_eq!(mem.entries()[1].role, Role::Assistant);
    }

    #[test]
    fn push_directly_with_memory_entry() {
        let mut mem = SessionMemory::new();
        let entry = MemoryEntry::new(Role::Assistant, "direct push");
        mem.push(entry);

        assert_eq!(mem.len(), 1);
        assert_eq!(mem.entries()[0].role, Role::Assistant);
        assert_eq!(mem.entries()[0].content, "direct push");
    }

    #[test]
    fn is_empty_on_new() {
        assert!(SessionMemory::new().is_empty());
    }

    #[test]
    fn clear_empties_memory() {
        let mut mem = SessionMemory::new();
        mem.add(Role::User, "hello");
        mem.clear();
        assert!(mem.is_empty());
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

        let entry = MemoryEntry::new(Role::User, "hello");
        let json = serde_json::to_string(&entry).unwrap();
        let e2: MemoryEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(e2.content, "hello");
        assert_eq!(e2.role, Role::User);
    }
}
