//! A scripted LLM stub for deterministic testing.
//!
//! [`StubClient`] returns pre-scripted [`LlmResponse`]s in order.
//! It returns an error if the script is exhausted, which signals that the
//! agent ran more steps than the test anticipated.

use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{anyhow, Result};
use async_trait::async_trait;

use crate::{LlmClient, LlmResponse, Message};

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
    async fn complete(&self, _system: Option<&str>, _messages: &[Message]) -> Result<LlmResponse> {
        let i = self.index.fetch_add(1, Ordering::SeqCst);
        self.responses.get(i).cloned().ok_or_else(|| {
            anyhow!("StubClient script exhausted — agent ran more steps than expected")
        })
    }
}
