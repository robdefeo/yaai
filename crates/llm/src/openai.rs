//! OpenAI chat completions client.

use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::mpsc;
use tracing::debug;

use crate::{ConversationTurn, LlmClient, LlmResponse, Message, ToolCall};

const OPENAI_CHAT_COMPLETIONS_URL: &str = "https://api.openai.com/v1/chat/completions";

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
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    stream: bool,
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

// ── Streaming SSE event types ────────────────────────────────────────────────

#[derive(Deserialize)]
struct ChatStreamResponse {
    choices: Vec<StreamChoice>,
}

#[derive(Deserialize)]
struct StreamChoice {
    delta: StreamDelta,
}

#[derive(Deserialize)]
struct StreamDelta {
    content: Option<String>,
    tool_calls: Option<Vec<OaiStreamToolCall>>,
}

#[derive(Deserialize)]
struct OaiStreamToolCall {
    id: Option<String>,
    function: Option<OaiStreamFunction>,
}

#[derive(Deserialize)]
struct OaiStreamFunction {
    name: Option<String>,
    arguments: Option<String>,
}

#[derive(Debug, Default)]
struct StreamAccumulator {
    text: String,
    tool_id: Option<String>,
    tool_name: Option<String>,
    tool_arguments: String,
}

// ─────────────────────────────────────────────────────────────────────────────

fn build_messages(system: Option<&str>, turns: &[ConversationTurn]) -> Vec<OaiMessage> {
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
    messages
}

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
            reasoning,
        } => OaiMessage {
            role: "assistant",
            content: reasoning.clone(),
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

fn pop_sse_line(buf: &mut Vec<u8>) -> Result<Option<String>> {
    let Some(pos) = buf.iter().position(|byte| *byte == b'\n') else {
        return Ok(None);
    };

    let mut line: Vec<u8> = buf.drain(..=pos).collect();
    if line.last() == Some(&b'\n') {
        line.pop();
    }
    if line.last() == Some(&b'\r') {
        line.pop();
    }

    String::from_utf8(line)
        .context("decoding SSE line from OpenAI stream")
        .map(Some)
}

fn apply_stream_data(
    data: &str,
    tx: &mpsc::UnboundedSender<String>,
    accumulator: &mut StreamAccumulator,
) -> Result<()> {
    if data == "[DONE]" {
        return Ok(());
    }

    let event: ChatStreamResponse =
        serde_json::from_str(data).context("parsing OpenAI stream event")?;

    for choice in event.choices {
        if let Some(text) = choice.delta.content {
            let _ = tx.send(text.clone());
            accumulator.text.push_str(&text);
        }

        if let Some(tool_calls) = choice.delta.tool_calls {
            for tool_call in tool_calls {
                if let Some(id) = tool_call.id {
                    accumulator.tool_id = Some(id);
                }
                if let Some(function) = tool_call.function {
                    if let Some(name) = function.name {
                        accumulator.tool_name = Some(name);
                    }
                    if let Some(arguments) = function.arguments {
                        accumulator.tool_arguments.push_str(&arguments);
                    }
                }
            }
        }
    }

    Ok(())
}

fn accumulated_response(accumulator: StreamAccumulator) -> Result<LlmResponse> {
    let tool_call = match (accumulator.tool_id, accumulator.tool_name) {
        (Some(id), Some(name)) => {
            let arguments = if accumulator.tool_arguments.is_empty() {
                serde_json::json!({})
            } else {
                serde_json::from_str(&accumulator.tool_arguments)
                    .context("parsing streamed tool call arguments")?
            };
            Some(ToolCall {
                id,
                name,
                arguments,
            })
        }
        _ => None,
    };

    let content = if accumulator.text.is_empty() {
        None
    } else {
        Some(accumulator.text)
    };

    Ok(LlmResponse { content, tool_call })
}

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
        let messages = build_messages(system, turns);

        let body = ChatRequest {
            model: &self.model,
            messages,
            tools,
            stream: false,
        };

        let response = self
            .client
            .post(OPENAI_CHAT_COMPLETIONS_URL)
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
                return Ok(LlmResponse {
                    content: msg.content, // preserve any reasoning alongside the tool call
                    tool_call: Some(ToolCall {
                        id: tc.id,
                        name: tc.function.name,
                        arguments: args,
                    }),
                });
            }
        }

        Ok(LlmResponse {
            content: msg.content,
            tool_call: None,
        })
    }

    async fn complete_streaming(
        &self,
        system: Option<&str>,
        turns: &[ConversationTurn],
        tools: &[Value],
        tx: &mpsc::UnboundedSender<String>,
    ) -> Result<LlmResponse> {
        debug!(model = %self.model, turns = turns.len(), "calling OpenAI (streaming)");

        let messages = build_messages(system, turns);
        let body = ChatRequest {
            model: &self.model,
            messages,
            tools,
            stream: true,
        };

        let mut response = self
            .client
            .post(OPENAI_CHAT_COMPLETIONS_URL)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .context("sending streaming request to OpenAI")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            bail!("OpenAI API error ({}): {}", status, body);
        }

        let mut buf = Vec::new();
        let mut accumulator = StreamAccumulator::default();

        while let Some(chunk) = response
            .chunk()
            .await
            .context("reading OpenAI SSE stream")?
        {
            buf.extend_from_slice(&chunk);

            while let Some(line) = pop_sse_line(&mut buf)? {
                let Some(data) = line.strip_prefix("data: ") else {
                    continue;
                };

                apply_stream_data(data, tx, &mut accumulator)?;
            }
        }

        accumulated_response(accumulator)
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
            reasoning: None,
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
            stream: false,
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
            stream: false,
        };
        let json = serde_json::to_value(&req).unwrap();
        assert!(json.get("tools").is_some());
    }

    #[test]
    fn stream_true_is_included_in_request_json() {
        let req = ChatRequest {
            model: "gpt-test",
            messages: vec![],
            tools: &[],
            stream: true,
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["stream"], true);
    }

    #[test]
    fn sse_line_buffer_preserves_utf8_split_across_chunks() {
        let line = "data: {\"choices\":[{\"delta\":{\"content\":\"hello 😀\"}}]}\n";
        let split = line.find('😀').unwrap() + 1;
        let bytes = line.as_bytes();
        let mut buf = Vec::new();

        buf.extend_from_slice(&bytes[..split]);
        assert_eq!(pop_sse_line(&mut buf).unwrap(), None);

        buf.extend_from_slice(&bytes[split..]);
        assert_eq!(
            pop_sse_line(&mut buf).unwrap().as_deref(),
            Some(line.trim_end_matches('\n'))
        );
    }

    #[test]
    fn stream_data_accumulates_text_and_emits_deltas() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let mut accumulator = StreamAccumulator::default();

        apply_stream_data(
            "{\"choices\":[{\"delta\":{\"content\":\"hello \"}}]}",
            &tx,
            &mut accumulator,
        )
        .unwrap();
        apply_stream_data(
            "{\"choices\":[{\"delta\":{\"content\":\"world\"}}]}",
            &tx,
            &mut accumulator,
        )
        .unwrap();

        assert_eq!(rx.try_recv().unwrap(), "hello ");
        assert_eq!(rx.try_recv().unwrap(), "world");
        let response = accumulated_response(accumulator).unwrap();
        assert_eq!(response.content.as_deref(), Some("hello world"));
        assert!(response.tool_call.is_none());
    }

    #[test]
    fn stream_data_accumulates_tool_call_arguments() {
        let (tx, _rx) = mpsc::unbounded_channel();
        let mut accumulator = StreamAccumulator::default();

        apply_stream_data(
            "{\"choices\":[{\"delta\":{\"tool_calls\":[{\"id\":\"call_01\",\"type\":\"function\",\"function\":{\"name\":\"read\",\"arguments\":\"{\\\"file_path\\\":\"}}]}}]}",
            &tx,
            &mut accumulator,
        )
        .unwrap();
        apply_stream_data(
            "{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"/tmp/a.txt\\\"}\"}}]}}]}",
            &tx,
            &mut accumulator,
        )
        .unwrap();

        let response = accumulated_response(accumulator).unwrap();
        assert!(response.content.is_none());
        let tool_call = response.tool_call.unwrap();
        assert_eq!(tool_call.id, "call_01");
        assert_eq!(tool_call.name, "read");
        assert_eq!(tool_call.arguments["file_path"], "/tmp/a.txt");
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

    #[test]
    fn tool_call_with_reasoning_sets_content_alongside_tool_calls() {
        let turn = ConversationTurn::AssistantToolCall {
            id: "call_01".to_string(),
            name: "read".to_string(),
            arguments: serde_json::json!({"file_path": "/tmp/a.txt"}),
            reasoning: Some("I should read the file.".to_string()),
        };
        let msg = turn_to_oai(&turn);
        assert_eq!(msg.role, "assistant");
        assert_eq!(msg.content.as_deref(), Some("I should read the file."));
        assert!(msg.tool_calls.is_some());
        assert_eq!(msg.tool_calls.unwrap()[0].id, "call_01");
    }

    #[test]
    fn tool_call_without_reasoning_has_no_content() {
        let turn = ConversationTurn::AssistantToolCall {
            id: "call_02".to_string(),
            name: "read".to_string(),
            arguments: serde_json::json!({}),
            reasoning: None,
        };
        let msg = turn_to_oai(&turn);
        assert!(msg.content.is_none());
        assert!(msg.tool_calls.is_some());
    }
}
