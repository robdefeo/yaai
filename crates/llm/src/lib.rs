//! LLM client abstraction.
//!
//! Defines the [`LlmClient`] trait and provides:
//! - [`StubClient`]: scripted responses for deterministic testing
//! - [`OpenAiClient`]: calls the OpenAI chat completions API
//! - [`ClaudeClient`]: calls the Anthropic Messages API

pub mod claude;
pub mod openai;
pub mod stub;

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

pub use claude::ClaudeClient;
pub use openai::OpenAiClient;
pub use stub::StubClient;

/// A message in the conversation history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".into(),
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".into(),
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".into(),
            content: content.into(),
        }
    }
}

/// A tool call emitted by the LLM.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCall {
    pub name: String,
    pub arguments: serde_json::Value,
}

/// The LLM's response to a completion request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmResponse {
    /// Free-text content (reasoning, final answer, etc.).
    pub content: Option<String>,
    /// Tool call, if the LLM chose to invoke a tool this step.
    pub tool_call: Option<ToolCall>,
}

impl LlmResponse {
    pub fn text(content: impl Into<String>) -> Self {
        Self {
            content: Some(content.into()),
            tool_call: None,
        }
    }

    pub fn tool(name: impl Into<String>, arguments: serde_json::Value) -> Self {
        Self {
            content: None,
            tool_call: Some(ToolCall {
                name: name.into(),
                arguments,
            }),
        }
    }

    /// True when the response contains text and no tool call — signals loop end.
    pub fn is_final_answer(&self) -> bool {
        self.tool_call.is_none() && self.content.is_some()
    }
}

/// Core LLM abstraction — send messages, receive a response.
///
/// `system` is passed separately so providers that require a dedicated
/// system field (e.g. Anthropic) can handle it natively, while others
/// (e.g. OpenAI) can prepend it to the messages array.
#[async_trait]
pub trait LlmClient: Send + Sync {
    async fn complete(&self, system: Option<&str>, messages: &[Message]) -> Result<LlmResponse>;
}

#[async_trait]
impl LlmClient for Box<dyn LlmClient> {
    async fn complete(&self, system: Option<&str>, messages: &[Message]) -> Result<LlmResponse> {
        (**self).complete(system, messages).await
    }
}
