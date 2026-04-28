//! LLM client abstraction.
//!
//! Defines the [`LlmClient`] trait and provides:
//! - [`StubClient`]: scripted responses for deterministic testing
//! - [`OpenAiClient`]: calls the OpenAI chat completions API
//! - [`AnthropicClient`]: calls the Anthropic Messages API

pub mod anthropic;
pub mod openai;
pub mod stub;

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::mpsc;

/// Build a [`reqwest::Client`] with sensible defaults: 10 s connect timeout,
/// 120 s overall request timeout.
pub(crate) fn default_http_client() -> reqwest::Client {
    reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(10))
        .timeout(std::time::Duration::from_secs(120))
        .build()
        .expect("failed to build reqwest client")
}

pub use anthropic::AnthropicClient;
pub use openai::OpenAiClient;
pub use stub::StubClient;

/// A plain text message in the conversation.
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

/// A structured conversation turn — text, a tool invocation, or a tool result.
///
/// Provider clients receive `&[ConversationTurn]` and are responsible for
/// serialising each variant into their own wire format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConversationTurn {
    Text(Message),
    AssistantToolCall {
        /// Provider-issued call ID used to correlate result messages.
        id: String,
        name: String,
        arguments: Value,
        /// Text the model emitted before choosing this tool call (chain-of-thought).
        reasoning: Option<String>,
    },
    ToolResult {
        tool_call_id: String,
        content: String,
    },
}

/// A tool call emitted by the LLM.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCall {
    /// Provider-issued call ID (e.g. `toolu_01…` for Anthropic, `call_01…` for OpenAI).
    pub id: String,
    pub name: String,
    pub arguments: Value,
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

    pub fn tool(id: impl Into<String>, name: impl Into<String>, arguments: Value) -> Self {
        Self {
            content: None,
            tool_call: Some(ToolCall {
                id: id.into(),
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

/// Core LLM abstraction — send a structured conversation, receive a response.
///
/// `system` is passed separately so providers that require a dedicated system
/// field (e.g. Anthropic) can handle it natively.
/// `tools` carries pre-formatted tool descriptors (Anthropic or OpenAI shape)
/// to include in the request.
#[async_trait]
pub trait LlmClient: Send + Sync {
    async fn complete(
        &self,
        system: Option<&str>,
        turns: &[ConversationTurn],
        tools: &[Value],
    ) -> Result<LlmResponse>;

    /// Stream token deltas to `tx` as they arrive, returning the full response
    /// once complete. The default implementation calls [`complete`] and emits
    /// the entire content as a single token.
    async fn complete_streaming(
        &self,
        system: Option<&str>,
        turns: &[ConversationTurn],
        tools: &[Value],
        tx: &mpsc::UnboundedSender<String>,
    ) -> Result<LlmResponse> {
        let response = self.complete(system, turns, tools).await?;
        if let Some(ref text) = response.content {
            let _ = tx.send(text.clone());
        }
        Ok(response)
    }
}

#[async_trait]
impl LlmClient for Box<dyn LlmClient> {
    async fn complete(
        &self,
        system: Option<&str>,
        turns: &[ConversationTurn],
        tools: &[Value],
    ) -> Result<LlmResponse> {
        (**self).complete(system, turns, tools).await
    }

    async fn complete_streaming(
        &self,
        system: Option<&str>,
        turns: &[ConversationTurn],
        tools: &[Value],
        tx: &mpsc::UnboundedSender<String>,
    ) -> Result<LlmResponse> {
        (**self).complete_streaming(system, turns, tools, tx).await
    }
}
