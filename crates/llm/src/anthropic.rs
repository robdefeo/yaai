//! Anthropic Claude Messages API client.

use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::debug;

use crate::{ConversationTurn, LlmClient, LlmResponse, Message};

const ANTHROPIC_API_URL: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION: &str = "2023-06-01";
const DEFAULT_MODEL: &str = "claude-sonnet-4-5";
/// Anthropic recommends setting max_tokens explicitly; use a generous default.
const DEFAULT_MAX_TOKENS: u32 = 4096;

#[derive(Debug, Clone)]
pub struct AnthropicClient {
    api_key: String,
    model: String,
    client: reqwest::Client,
}

impl AnthropicClient {
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
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "<[_]>::is_empty")]
    tools: &'a [Value],
}

/// A single message in the Anthropic wire format. Content is always a list of
/// blocks — even for plain text — to support multi-part assistant/tool turns.
#[derive(Serialize)]
struct AnthropicMessage {
    role: &'static str,
    content: Vec<AnthropicBlock>,
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    ToolResult {
        tool_use_id: String,
        content: String,
    },
}

#[derive(Deserialize)]
struct MessagesResponse {
    content: Vec<ContentBlock>,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ContentBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    #[serde(other)]
    Unknown,
}

// ─────────────────────────────────────────────────────────────────────────────

fn turn_to_anthropic(turn: &ConversationTurn) -> AnthropicMessage {
    match turn {
        ConversationTurn::Text(Message { role, content }) => AnthropicMessage {
            role: if role == "assistant" {
                "assistant"
            } else {
                "user"
            },
            content: vec![AnthropicBlock::Text {
                text: content.clone(),
            }],
        },
        ConversationTurn::AssistantToolCall {
            id,
            name,
            arguments,
        } => AnthropicMessage {
            role: "assistant",
            content: vec![AnthropicBlock::ToolUse {
                id: id.clone(),
                name: name.clone(),
                input: arguments.clone(),
            }],
        },
        ConversationTurn::ToolResult {
            tool_call_id,
            content,
        } => AnthropicMessage {
            role: "user",
            content: vec![AnthropicBlock::ToolResult {
                tool_use_id: tool_call_id.clone(),
                content: content.clone(),
            }],
        },
    }
}

// grcov-excl-start: real HTTP transport requires integration tests or an injected client seam
#[async_trait]
impl LlmClient for AnthropicClient {
    async fn complete(
        &self,
        system: Option<&str>,
        turns: &[ConversationTurn],
        tools: &[Value],
    ) -> Result<LlmResponse> {
        debug!(model = %self.model, turns = turns.len(), "calling Anthropic");

        let messages: Vec<AnthropicMessage> = turns.iter().map(turn_to_anthropic).collect();

        let body = MessagesRequest {
            model: &self.model,
            max_tokens: DEFAULT_MAX_TOKENS,
            system,
            messages,
            tools,
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
            if let ContentBlock::ToolUse { id, name, input } = block {
                return Ok(LlmResponse::tool(id.clone(), name.clone(), input.clone()));
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

// grcov-excl-start: exclude inline unit tests from production coverage
#[cfg(test)]
mod tests {
    use super::*;

    fn text_turn(role: &str, content: &str) -> ConversationTurn {
        ConversationTurn::Text(Message {
            role: role.to_string(),
            content: content.to_string(),
        })
    }

    #[test]
    fn user_text_turn_maps_to_user_role_with_text_block() {
        let msg = turn_to_anthropic(&text_turn("user", "hello"));
        assert_eq!(msg.role, "user");
        assert_eq!(msg.content.len(), 1);
        let json = serde_json::to_value(&msg.content[0]).unwrap();
        assert_eq!(json["type"], "text");
        assert_eq!(json["text"], "hello");
    }

    #[test]
    fn assistant_text_turn_maps_to_assistant_role() {
        let msg = turn_to_anthropic(&text_turn("assistant", "done"));
        assert_eq!(msg.role, "assistant");
        let json = serde_json::to_value(&msg.content[0]).unwrap();
        assert_eq!(json["type"], "text");
        assert_eq!(json["text"], "done");
    }

    #[test]
    fn system_text_turn_falls_through_to_user_role() {
        // Any non-"assistant" role maps to "user" at the wire level.
        let msg = turn_to_anthropic(&text_turn("system", "sys prompt"));
        assert_eq!(msg.role, "user");
    }

    #[test]
    fn assistant_tool_call_maps_to_tool_use_block() {
        let turn = ConversationTurn::AssistantToolCall {
            id: "toolu_01".to_string(),
            name: "read".to_string(),
            arguments: serde_json::json!({"file_path": "/tmp/a.txt"}),
        };
        let msg = turn_to_anthropic(&turn);
        assert_eq!(msg.role, "assistant");
        let json = serde_json::to_value(&msg.content[0]).unwrap();
        assert_eq!(json["type"], "tool_use");
        assert_eq!(json["id"], "toolu_01");
        assert_eq!(json["name"], "read");
        assert_eq!(json["input"]["file_path"], "/tmp/a.txt");
    }

    #[test]
    fn tool_result_maps_to_user_role_with_tool_result_block() {
        let turn = ConversationTurn::ToolResult {
            tool_call_id: "toolu_01".to_string(),
            content: "file contents here".to_string(),
        };
        let msg = turn_to_anthropic(&turn);
        assert_eq!(msg.role, "user");
        let json = serde_json::to_value(&msg.content[0]).unwrap();
        assert_eq!(json["type"], "tool_result");
        assert_eq!(json["tool_use_id"], "toolu_01");
        assert_eq!(json["content"], "file contents here");
    }

    #[test]
    fn empty_tools_omitted_from_request_json() {
        let req = MessagesRequest {
            model: "claude-test",
            max_tokens: 100,
            system: None,
            messages: vec![],
            tools: &[],
        };
        let json = serde_json::to_value(&req).unwrap();
        assert!(json.get("tools").is_none(), "empty tools must be omitted");
    }

    #[test]
    fn non_empty_tools_included_in_request_json() {
        let tools = vec![serde_json::json!({"type": "function", "name": "read"})];
        let req = MessagesRequest {
            model: "claude-test",
            max_tokens: 100,
            system: None,
            messages: vec![],
            tools: &tools,
        };
        let json = serde_json::to_value(&req).unwrap();
        assert!(json.get("tools").is_some());
    }
}
// grcov-excl-stop
