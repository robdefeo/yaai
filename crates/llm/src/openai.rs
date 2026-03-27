//! OpenAI chat completions client.

use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::{LlmClient, LlmResponse, Message};

#[derive(Debug, Clone)]
pub struct OpenAiClient {
    api_key: String,
    model: String,
    client: reqwest::Client,
}

impl OpenAiClient {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: model.into(),
            client: reqwest::Client::new(),
        }
    }
}

// ── OpenAI wire types ────────────────────────────────────────────────────────

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: &'a [Message],
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: ResponseMessage,
}

#[derive(Deserialize)]
struct ResponseMessage {
    content: Option<String>,
    tool_calls: Option<Vec<OaiToolCall>>,
}

#[derive(Deserialize)]
struct OaiToolCall {
    function: OaiFunction,
}

#[derive(Deserialize)]
struct OaiFunction {
    name: String,
    /// JSON-encoded string of the arguments object.
    arguments: String,
}

// ─────────────────────────────────────────────────────────────────────────────

// grcov-excl-start
#[async_trait]
impl LlmClient for OpenAiClient {
    async fn complete(&self, system: Option<&str>, messages: &[Message]) -> Result<LlmResponse> {
        debug!(model = %self.model, messages = messages.len(), "calling OpenAI");

        // OpenAI expects the system prompt as the first entry in the messages array.
        let mut full_messages;
        let messages = if let Some(sys) = system {
            full_messages = Vec::with_capacity(messages.len() + 1);
            full_messages.push(Message::system(sys));
            full_messages.extend_from_slice(messages);
            full_messages.as_slice()
        } else {
            messages
        };

        let body = ChatRequest {
            model: &self.model,
            messages,
        };

        let response = self
            .client
            .post("https://api.openai.com/v1/chat/completions")
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .context("sending request to OpenAI")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            bail!("OpenAI API error ({}): {}", status, body);
        }

        let resp: ChatResponse = response
            .json()
            .await
            .context("parsing OpenAI response")?;

        let msg = resp
            .choices
            .into_iter()
            .next()
            .context("no choices in OpenAI response")?
            .message;

        if let Some(tool_calls) = msg.tool_calls {
            if let Some(tc) = tool_calls.into_iter().next() {
                let args: serde_json::Value = serde_json::from_str(&tc.function.arguments)
                    .context("parsing tool call arguments")?;
                return Ok(LlmResponse::tool(tc.function.name, args));
            }
        }

        Ok(LlmResponse {
            content: msg.content,
            tool_call: None,
        })
    }
}
// grcov-excl-stop