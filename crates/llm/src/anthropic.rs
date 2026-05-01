//! Anthropic Claude Messages API client.

use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::mpsc;
use tracing::{debug, warn};

use crate::{sse::pop_sse_line, ConversationTurn, LlmClient, LlmResponse, Message, ToolCall};

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
struct ToolChoice {
    #[serde(rename = "type")]
    kind: &'static str,
    disable_parallel_tool_use: bool,
}

#[derive(Serialize)]
struct MessagesRequest<'a> {
    model: &'a str,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<&'a str>,
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "<[_]>::is_empty")]
    tools: &'a [Value],
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    stream: bool,
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

// ── Streaming SSE event types ────────────────────────────────────────────────

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum StreamEvent {
    ContentBlockStart {
        index: usize,
        content_block: StreamBlock,
    },
    ContentBlockDelta {
        index: usize,
        delta: StreamDelta,
    },
    Error {
        error: StreamApiError,
    },
    #[serde(other)]
    Other,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum StreamBlock {
    Text {
        #[allow(dead_code)]
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
    },
    #[serde(other)]
    Other,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum StreamDelta {
    TextDelta {
        text: String,
    },
    InputJsonDelta {
        partial_json: String,
    },
    #[serde(other)]
    Other,
}

#[derive(Deserialize)]
struct StreamApiError {
    message: String,
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
            reasoning,
        } => {
            let mut blocks = Vec::new();
            if let Some(text) = reasoning {
                blocks.push(AnthropicBlock::Text { text: text.clone() });
            }
            blocks.push(AnthropicBlock::ToolUse {
                id: id.clone(),
                name: name.clone(),
                input: arguments.clone(),
            });
            AnthropicMessage {
                role: "assistant",
                content: blocks,
            }
        }
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

/// Parse a slice of [`ContentBlock`]s into an [`LlmResponse`].
///
/// Text blocks are concatenated (newline-separated) so that multiple `Text`
/// blocks before a `ToolUse` are all preserved as reasoning. The first
/// `ToolUse` block terminates the scan and the accumulated text is returned
/// as `content` alongside the tool call.
fn parse_blocks(blocks: &[ContentBlock]) -> LlmResponse {
    let mut reasoning: Option<String> = None;
    for block in blocks {
        match block {
            ContentBlock::Text { text } => match reasoning.as_mut() {
                Some(r) => {
                    r.push('\n');
                    r.push_str(text);
                }
                None => reasoning = Some(text.clone()),
            },
            ContentBlock::ToolUse { id, name, input } => {
                return LlmResponse {
                    content: reasoning,
                    tool_call: Some(ToolCall {
                        id: id.clone(),
                        name: name.clone(),
                        arguments: input.clone(),
                    }),
                };
            }
            ContentBlock::Unknown => {}
        }
    }
    if let Some(text) = reasoning {
        return LlmResponse::text(text);
    }
    LlmResponse {
        content: None,
        tool_call: None,
    }
}

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

        let tool_choice = (!tools.is_empty()).then_some(ToolChoice {
            kind: "auto",
            disable_parallel_tool_use: true,
        });
        let body = MessagesRequest {
            model: &self.model,
            max_tokens: DEFAULT_MAX_TOKENS,
            system,
            messages,
            tools,
            tool_choice,
            stream: false,
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

        Ok(parse_blocks(&resp.content))
    }

    async fn complete_streaming(
        &self,
        system: Option<&str>,
        turns: &[ConversationTurn],
        tools: &[Value],
        tx: &mpsc::UnboundedSender<String>,
    ) -> Result<LlmResponse> {
        debug!(model = %self.model, turns = turns.len(), "calling Anthropic (streaming)");

        let messages: Vec<AnthropicMessage> = turns.iter().map(turn_to_anthropic).collect();

        let tool_choice = (!tools.is_empty()).then_some(ToolChoice {
            kind: "auto",
            disable_parallel_tool_use: true,
        });
        let body = MessagesRequest {
            model: &self.model,
            max_tokens: DEFAULT_MAX_TOKENS,
            system,
            messages,
            tools,
            tool_choice,
            stream: true,
        };

        let mut response = self
            .client
            .post(ANTHROPIC_API_URL)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .json(&body)
            .send()
            .await
            .context("sending streaming request to Anthropic")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            bail!("Anthropic API error ({}): {}", status, body);
        }

        let mut buf = Vec::new();
        let mut text = String::new();
        let mut tool_id: Option<String> = None;
        let mut tool_name: Option<String> = None;
        let mut tool_json = String::new();
        let mut tool_block_index: Option<usize> = None;

        while let Some(chunk) = response.chunk().await.context("reading SSE stream")? {
            buf.extend_from_slice(&chunk);

            while let Some(line) = pop_sse_line(&mut buf)? {
                let Some(data) = line.strip_prefix("data: ") else {
                    continue;
                };

                match serde_json::from_str::<StreamEvent>(data) {
                    Ok(StreamEvent::ContentBlockStart {
                        index,
                        content_block,
                    }) => {
                        if let StreamBlock::ToolUse { id, name } = content_block {
                            if tool_id.is_none() {
                                tool_id = Some(id);
                                tool_name = Some(name);
                                tool_block_index = Some(index);
                            }
                        }
                    }
                    Ok(StreamEvent::ContentBlockDelta { index, delta }) => match delta {
                        StreamDelta::TextDelta { text: t } => {
                            let _ = tx.send(t.clone());
                            text.push_str(&t);
                        }
                        StreamDelta::InputJsonDelta { partial_json } => {
                            if Some(index) == tool_block_index {
                                tool_json.push_str(&partial_json);
                            }
                        }
                        StreamDelta::Other => {}
                    },
                    Ok(StreamEvent::Error { error }) => {
                        bail!("Anthropic stream error: {}", error.message);
                    }
                    Ok(StreamEvent::Other) => {}
                    Err(e) => {
                        warn!(error = %e, raw = %data, "failed to parse SSE event");
                    }
                }
            }
        }

        let tool_call = match (tool_id, tool_name) {
            (Some(id), Some(name)) => {
                let arguments = if tool_json.is_empty() {
                    serde_json::json!({})
                } else {
                    serde_json::from_str(&tool_json)
                        .with_context(|| format!("parsing tool input JSON: {:?}", tool_json))?
                };
                Some(ToolCall {
                    id,
                    name,
                    arguments,
                })
            }
            _ => None,
        };

        let content = if text.is_empty() { None } else { Some(text) };

        Ok(LlmResponse { content, tool_call })
    }
}

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
            reasoning: None,
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
            tool_choice: None,
            stream: false,
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
            tool_choice: None,
            stream: false,
        };
        let json = serde_json::to_value(&req).unwrap();
        assert!(json.get("tools").is_some());
    }

    #[test]
    fn tool_call_with_reasoning_emits_text_then_tool_use_blocks() {
        let turn = ConversationTurn::AssistantToolCall {
            id: "toolu_01".to_string(),
            name: "read".to_string(),
            arguments: serde_json::json!({"file_path": "/tmp/a.txt"}),
            reasoning: Some("I should read the file first.".to_string()),
        };
        let msg = turn_to_anthropic(&turn);
        assert_eq!(msg.role, "assistant");
        assert_eq!(msg.content.len(), 2);
        let first = serde_json::to_value(&msg.content[0]).unwrap();
        assert_eq!(first["type"], "text");
        assert_eq!(first["text"], "I should read the file first.");
        let second = serde_json::to_value(&msg.content[1]).unwrap();
        assert_eq!(second["type"], "tool_use");
        assert_eq!(second["id"], "toolu_01");
    }

    #[test]
    fn complete_text_then_tool_use_returns_reasoning_with_tool_call() {
        let blocks = vec![
            ContentBlock::Text {
                text: "Let me check that file.".to_string(),
            },
            ContentBlock::ToolUse {
                id: "toolu_01".to_string(),
                name: "read".to_string(),
                input: serde_json::json!({"file_path": "/tmp/a.txt"}),
            },
        ];
        let r = parse_blocks(&blocks);
        assert_eq!(r.content.as_deref(), Some("Let me check that file."));
        let tc = r.tool_call.unwrap();
        assert_eq!(tc.id, "toolu_01");
        assert_eq!(tc.name, "read");
    }

    #[test]
    fn complete_tool_use_only_returns_no_reasoning() {
        let blocks = vec![ContentBlock::ToolUse {
            id: "toolu_02".to_string(),
            name: "read".to_string(),
            input: serde_json::json!({}),
        }];
        let r = parse_blocks(&blocks);
        assert!(r.content.is_none());
        assert!(r.tool_call.is_some());
    }

    #[test]
    fn multiple_text_blocks_are_concatenated_as_reasoning() {
        let blocks = vec![
            ContentBlock::Text {
                text: "First thought.".to_string(),
            },
            ContentBlock::Text {
                text: "Second thought.".to_string(),
            },
            ContentBlock::ToolUse {
                id: "toolu_03".to_string(),
                name: "read".to_string(),
                input: serde_json::json!({}),
            },
        ];
        let r = parse_blocks(&blocks);
        assert_eq!(
            r.content.as_deref(),
            Some("First thought.\nSecond thought.")
        );
        assert!(r.tool_call.is_some());
    }

    #[test]
    fn unknown_block_alone_returns_empty_response() {
        let r = parse_blocks(&[ContentBlock::Unknown]);
        assert!(r.content.is_none());
        assert!(r.tool_call.is_none());
    }

    #[test]
    fn unknown_block_mixed_with_text_is_skipped() {
        let blocks = vec![
            ContentBlock::Text {
                text: "thought".to_string(),
            },
            ContentBlock::Unknown,
        ];
        let r = parse_blocks(&blocks);
        assert_eq!(r.content.as_deref(), Some("thought"));
        assert!(r.tool_call.is_none());
    }

    #[test]
    fn unknown_block_before_tool_use_is_skipped() {
        let blocks = vec![
            ContentBlock::Unknown,
            ContentBlock::ToolUse {
                id: "toolu_01".to_string(),
                name: "read".to_string(),
                input: serde_json::json!({}),
            },
        ];
        let r = parse_blocks(&blocks);
        assert!(r.content.is_none());
        assert!(r.tool_call.is_some());
    }

    // ── Streaming accumulation tests ─────────────────────────────────────────

    /// Simulate the streaming loop over a set of SSE data lines and return the
    /// accumulated tool call id, name, and raw json string.  The mpsc channel is
    /// needed because `TextDelta` events send tokens to the UI.
    fn run_sse_accumulation(sse_lines: &[&str]) -> (Option<String>, Option<String>, String) {
        let (_tx, _rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        let mut tool_id: Option<String> = None;
        let mut tool_name: Option<String> = None;
        let mut tool_json = String::new();
        let mut tool_block_index: Option<usize> = None;

        for line in sse_lines {
            let Some(data) = line.strip_prefix("data: ") else {
                continue;
            };
            let Ok(event) = serde_json::from_str::<StreamEvent>(data) else {
                continue;
            };
            match event {
                StreamEvent::ContentBlockStart {
                    index,
                    content_block: StreamBlock::ToolUse { id, name },
                } => {
                    if tool_id.is_none() {
                        tool_id = Some(id);
                        tool_name = Some(name);
                        tool_block_index = Some(index);
                    }
                }
                StreamEvent::ContentBlockStart { .. } => {}
                StreamEvent::ContentBlockDelta { index, delta } => match delta {
                    StreamDelta::TextDelta { .. } => {}
                    StreamDelta::InputJsonDelta { partial_json } => {
                        if Some(index) == tool_block_index {
                            tool_json.push_str(&partial_json);
                        }
                    }
                    StreamDelta::Other => {}
                },
                _ => {}
            }
        }
        (tool_id, tool_name, tool_json)
    }

    #[test]
    fn single_tool_call_accumulates_correctly() {
        let lines = [
            r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_01","name":"list_dir"}}"#,
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"dir_path\":"}}"#,
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"\"/\"}"}}"#,
        ];
        let (id, name, json) = run_sse_accumulation(&lines);
        assert_eq!(id.as_deref(), Some("toolu_01"));
        assert_eq!(name.as_deref(), Some("list_dir"));
        let args: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(args["dir_path"], "/");
    }

    #[test]
    fn parallel_tool_calls_captures_only_first() {
        // Two sequential tool_use blocks in the same response (parallel tool use).
        // Only the first should be captured; the second's JSON must NOT contaminate
        // tool_json (which would produce invalid concatenated JSON).
        let lines = [
            r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_01","name":"list_dir"}}"#,
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"dir_path\":\"/\"}"}}"#,
            r#"data: {"type":"content_block_stop","index":0}"#,
            r#"data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_02","name":"read"}}"#,
            r#"data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"file_path\":\"/README.md\"}"}}"#,
            r#"data: {"type":"content_block_stop","index":1}"#,
        ];
        let (id, name, json) = run_sse_accumulation(&lines);
        assert_eq!(id.as_deref(), Some("toolu_01"));
        assert_eq!(name.as_deref(), Some("list_dir"));
        // json must be valid and contain only the first tool's input
        let args: serde_json::Value =
            serde_json::from_str(&json).expect("tool_json must be valid JSON");
        assert_eq!(args["dir_path"], "/");
        assert!(args.get("file_path").is_none());
    }
}
