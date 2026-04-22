use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use serde_json::Value;

use crate::{ConversationTurn, LlmClient, LlmResponse};

pub struct StubClient {
    responses: Vec<LlmResponse>,
    index: AtomicUsize,
}

impl StubClient {
    /// Create a stub with a sequence of responses (first element = first returned).
    pub fn new(responses: Vec<LlmResponse>) -> Self {
        Self {
            responses,
            index: AtomicUsize::new(0),
        }
    }
}

#[async_trait]
impl LlmClient for StubClient {
    async fn complete(
        &self,
        _system: Option<&str>,
        _turns: &[ConversationTurn],
        _tools: &[Value],
    ) -> Result<LlmResponse> {
        let i = self.index.fetch_add(1, Ordering::SeqCst);
        self.responses.get(i).cloned().ok_or_else(|| {
            anyhow!("StubClient script exhausted — agent ran more steps than expected")
        })
    }
}
