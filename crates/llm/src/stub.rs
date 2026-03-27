//! A scripted LLM stub for deterministic testing.
//!
//! [`StubClient`] returns pre-scripted [`LlmResponse`]s in order.
//! It returns an error if the script is exhausted, which signals that the
//! agent ran more steps than the test anticipated.

use std::sync::Mutex;

use anyhow::{anyhow, Result};
use async_trait::async_trait;

use crate::{LlmClient, LlmResponse, Message};

pub struct StubClient {
    script: Mutex<Vec<LlmResponse>>,
}

impl StubClient {
    /// Create a stub with a sequence of responses (first element = first returned).
    pub fn new(mut responses: Vec<LlmResponse>) -> Self {
        responses.reverse(); // pop() from the end returns in original order
        Self {
            script: Mutex::new(responses),
        }
    }
}

#[async_trait]
impl LlmClient for StubClient {
    async fn complete(&self, _system: Option<&str>, _messages: &[Message]) -> Result<LlmResponse> {
        self.script.lock().unwrap().pop().ok_or_else(|| {
            anyhow!("StubClient script exhausted — agent ran more steps than expected")
        })
    }
}
