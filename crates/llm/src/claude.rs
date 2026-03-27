//! Anthropic Claude Messages API client.

use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::{LlmClient, LlmResponse, Message};

const ANTHROPIC_API_URL: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION: &str = "2023-06-01";
const DEFAULT_MODEL: &str = "claude-sonnet-4-5";
/// Anthropic recommends setting max_tokens explicitly; use a generous default.
const DEFAULT_MAX_TOKENS: u32 = 4096;

#[derive(Debug, Clone)]
pub struct ClaudeClient {
    api_key: String,
    model: String,
    client: reqwest::Client,
}

impl ClaudeClient {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self::with_client(api_key, model, crate::default_http_client())
    }

    /// Construct with a caller-supplied [`reqwest::Client`] for full control
    /// over timeouts, proxies, TLS, etc.
    pub fn with_client(
        api_key: impl Into<String>,
        model: impl Into<String>,
        client: reqwest::Client,
    ) -> Self {
        let model = model.into();
        let model = if model.is_empty() {
            DEFAULT_MODEL.to_string()
        } else {
            model
        };
        Self {
            api_key: api_key.into(),
            model,
            client,
        }
    }
}

// ── Anthropic wire types ─────────────────────────────────────────────────────

#[derive(Serialize)]
struct MessagesRequest<'a> {
    model: &'a str,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<&'a str>,
    messages: &'a [Message],
}

#[derive(Deserialize)]
struct MessagesResponse {
    content: Vec<ContentBlock>,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ContentBlock {
    Text { text: String },
    ToolUse { name: String, input: serde_json::Value },
    #[serde(other)]
    Unknown,
}

// ─────────────────────────────────────────────────────────────────────────────

// grcov-excl-start
#[async_trait]
impl LlmClient for ClaudeClient {
    async fn complete(&self, system: Option<&str>, messages: &[Message]) -> Result<LlmResponse> {
        debug!(model = %self.model, messages = messages.len(), "calling Claude");

        let body = MessagesRequest {
            model: &self.model,
            max_tokens: DEFAULT_MAX_TOKENS,
            system,
            messages,
        };

        let response = self
            .client
            .post(ANTHROPIC_API_URL)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .json(&body)
            .send()
            .await
            .context("sending request to Anthropic")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            bail!("Anthropic API error ({}): {}", status, body);
        }

        let resp: MessagesResponse = response
            .json()
            .await
            .context("parsing Anthropic response")?;

        // Prefer tool_use blocks first (same priority as OpenAI implementation).
        for block in &resp.content {
            if let ContentBlock::ToolUse { name, input } = block {
                return Ok(LlmResponse::tool(name.clone(), input.clone()));
            }
        }

        // Fall back to the first text block.
        for block in resp.content {
            if let ContentBlock::Text { text } = block {
                return Ok(LlmResponse::text(text));
            }
        }

        Ok(LlmResponse {
            content: None,
            tool_call: None,
        })
    }
}
// grcov-excl-stop