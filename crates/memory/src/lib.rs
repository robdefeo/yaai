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
/// Entries accumulate in insertion order. Use [`SessionMemory::compact`] to
/// trim history when approaching context limits.
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

    /// Estimated token count (rough: 1 token ≈ 4 chars).
    pub fn estimated_tokens(&self) -> usize {
        self.entries.iter().map(|e| e.content.len() / 4).sum()
    }

    /// Remove all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Compact: keep any leading System entries and the last `keep_last` entries,
    /// replacing the middle with a summary marker.
    ///
    /// This is a simple sliding-window compaction; a richer implementation would
    /// call an LLM to summarise the dropped section.
    pub fn compact(&mut self, keep_last: usize) {
        let system_head: Vec<MemoryEntry> = self
            .entries
            .iter()
            .take_while(|e| e.role == Role::System)
            .cloned()
            .collect();

        let rest: Vec<MemoryEntry> = self
            .entries
            .iter()
            .skip(system_head.len())
            .cloned()
            .collect();

        if rest.len() <= keep_last {
            return;
        }

        let dropped = rest.len() - keep_last;
        let kept_tail = rest[rest.len() - keep_last..].to_vec();
        let summary = MemoryEntry::new(
            Role::Assistant,
            format!("[{dropped} earlier messages compacted]"),
        );

        self.entries = system_head;
        self.entries.push(summary);
        self.entries.extend(kept_tail);
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
    fn estimated_tokens_grows_with_content() {
        let mut mem = SessionMemory::new();
        mem.add(Role::User, "hello world this is a test message");
        assert!(mem.estimated_tokens() > 0);
    }

    #[test]
    fn compact_keeps_system_and_tail() {
        let mut mem = SessionMemory::new();
        mem.add(Role::System, "you are a helpful agent");
        for i in 0..10 {
            mem.add(Role::User, format!("message {i}"));
            mem.add(Role::Assistant, format!("reply {i}"));
        }

        mem.compact(4);

        assert_eq!(mem.entries()[0].role, Role::System);
        assert!(mem.entries().iter().any(|e| e.content.contains("compacted")));
        let tail = &mem.entries()[mem.len() - 4..];
        assert_eq!(tail.len(), 4);
    }

    #[test]
    fn compact_noop_when_short() {
        let mut mem = SessionMemory::new();
        mem.add(Role::User, "a");
        mem.add(Role::User, "b");
        let original_len = mem.len();
        mem.compact(10);
        assert_eq!(mem.len(), original_len);
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
