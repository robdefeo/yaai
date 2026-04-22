//! OpenAI chat completions client.

use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::debug;

use crate::{ConversationTurn, LlmClient, LlmResponse, Message};

#[derive(Debug, Clone)]
pub struct OpenAiClient {
    api_key: String,
    model: String,
    client: reqwest::Client,
}

impl OpenAiClient {
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
        Self {
            api_key: api_key.into(),
            model: model.into(),
            client,
        }
    }
}

// ── OpenAI wire types ────────────────────────────────────────────────────────

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<OaiMessage>,
    #[serde(skip_serializing_if = "<[_]>::is_empty")]
    tools: &'a [Value],
}

/// A single message in the OpenAI wire format.
#[derive(Serialize)]
struct OaiMessage {
    role: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    /// Present only on assistant messages that made tool calls.
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OaiOutboundToolCall>>,
    /// Present only on tool-result messages.
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Serialize)]
struct OaiOutboundToolCall {
    id: String,
    #[serde(rename = "type")]
    kind: &'static str,
    function: OaiOutboundFunction,
}

#[derive(Serialize)]
struct OaiOutboundFunction {
    name: String,
    arguments: String,
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
    tool_calls: Option<Vec<OaiInboundToolCall>>,
}

#[derive(Deserialize)]
struct OaiInboundToolCall {
    id: String,
    function: OaiInboundFunction,
}

#[derive(Deserialize)]
struct OaiInboundFunction {
    name: String,
    /// JSON-encoded string of the arguments object.
    arguments: String,
}

// ─────────────────────────────────────────────────────────────────────────────

fn turn_to_oai(turn: &ConversationTurn) -> OaiMessage {
    match turn {
        ConversationTurn::Text(Message { role, content }) => OaiMessage {
            role: if role == "assistant" {
                "assistant"
            } else {
                "user"
            },
            content: Some(content.clone()),
            tool_calls: None,
            tool_call_id: None,
        },
        ConversationTurn::AssistantToolCall {
            id,
            name,
            arguments,
        } => OaiMessage {
            role: "assistant",
            content: None,
            tool_calls: Some(vec![OaiOutboundToolCall {
                id: id.clone(),
                kind: "function",
                function: OaiOutboundFunction {
                    name: name.clone(),
                    arguments: arguments.to_string(),
                },
            }]),
            tool_call_id: None,
        },
        ConversationTurn::ToolResult {
            tool_call_id,
            content,
        } => OaiMessage {
            role: "tool",
            content: Some(content.clone()),
            tool_calls: None,
            tool_call_id: Some(tool_call_id.clone()),
        },
    }
}

// grcov-excl-start: real HTTP transport requires integration tests or an injected client seam
#[async_trait]
impl LlmClient for OpenAiClient {
    async fn complete(
        &self,
        system: Option<&str>,
        turns: &[ConversationTurn],
        tools: &[Value],
    ) -> Result<LlmResponse> {
        debug!(model = %self.model, turns = turns.len(), "calling OpenAI");

        // OpenAI expects the system prompt as the first entry in the messages array.
        let mut messages: Vec<OaiMessage> = turns.iter().map(turn_to_oai).collect();
        if let Some(sys) = system {
            messages.insert(
                0,
                OaiMessage {
                    role: "system",
                    content: Some(sys.to_string()),
                    tool_calls: None,
                    tool_call_id: None,
                },
            );
        }

        let body = ChatRequest {
            model: &self.model,
            messages,
            tools,
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

        let resp: ChatResponse = response.json().await.context("parsing OpenAI response")?;

        let msg = resp
            .choices
            .into_iter()
            .next()
            .context("no choices in OpenAI response")?
            .message;

        if let Some(tool_calls) = msg.tool_calls {
            if let Some(tc) = tool_calls.into_iter().next() {
                let args: Value = serde_json::from_str(&tc.function.arguments)
                    .context("parsing tool call arguments")?;
                return Ok(LlmResponse::tool(tc.id, tc.function.name, args));
            }
        }

        Ok(LlmResponse {
            content: msg.content,
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
    fn user_text_turn_maps_to_user_role() {
        let msg = turn_to_oai(&text_turn("user", "hello"));
        assert_eq!(msg.role, "user");
        assert_eq!(msg.content.as_deref(), Some("hello"));
        assert!(msg.tool_calls.is_none());
        assert!(msg.tool_call_id.is_none());
    }

    #[test]
    fn assistant_text_turn_maps_to_assistant_role() {
        let msg = turn_to_oai(&text_turn("assistant", "done"));
        assert_eq!(msg.role, "assistant");
        assert_eq!(msg.content.as_deref(), Some("done"));
    }

    #[test]
    fn system_text_turn_falls_through_to_user_role() {
        let msg = turn_to_oai(&text_turn("system", "sys"));
        assert_eq!(msg.role, "user");
    }

    #[test]
    fn assistant_tool_call_serializes_to_tool_calls_array() {
        let turn = ConversationTurn::AssistantToolCall {
            id: "call_01".to_string(),
            name: "read".to_string(),
            arguments: serde_json::json!({"file_path": "/tmp/a.txt"}),
        };
        let msg = turn_to_oai(&turn);
        assert_eq!(msg.role, "assistant");
        assert!(msg.content.is_none());
        assert!(msg.tool_call_id.is_none());

        let calls = msg.tool_calls.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_01");
        assert_eq!(calls[0].kind, "function");
        assert_eq!(calls[0].function.name, "read");
        // arguments must be a JSON string (OpenAI wire format)
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["file_path"], "/tmp/a.txt");
    }

    #[test]
    fn tool_result_maps_to_tool_role_with_tool_call_id() {
        let turn = ConversationTurn::ToolResult {
            tool_call_id: "call_01".to_string(),
            content: "file contents".to_string(),
        };
        let msg = turn_to_oai(&turn);
        assert_eq!(msg.role, "tool");
        assert_eq!(msg.content.as_deref(), Some("file contents"));
        assert_eq!(msg.tool_call_id.as_deref(), Some("call_01"));
        assert!(msg.tool_calls.is_none());
    }

    #[test]
    fn empty_tools_omitted_from_request_json() {
        let req = ChatRequest {
            model: "gpt-test",
            messages: vec![],
            tools: &[],
        };
        let json = serde_json::to_value(&req).unwrap();
        assert!(json.get("tools").is_none(), "empty tools must be omitted");
    }

    #[test]
    fn non_empty_tools_included_in_request_json() {
        let tools = vec![serde_json::json!({"type": "function", "name": "read"})];
        let req = ChatRequest {
            model: "gpt-test",
            messages: vec![],
            tools: &tools,
        };
        let json = serde_json::to_value(&req).unwrap();
        assert!(json.get("tools").is_some());
    }

    #[test]
    fn optional_fields_omitted_when_none() {
        let msg = OaiMessage {
            role: "user",
            content: Some("hi".to_string()),
            tool_calls: None,
            tool_call_id: None,
        };
        let json = serde_json::to_value(&msg).unwrap();
        assert!(json.get("tool_calls").is_none());
        assert!(json.get("tool_call_id").is_none());
    }
}
// grcov-excl-stop
