use yaai_llm::{LlmClient, LlmResponse, Message, StubClient, ToolCall};

#[tokio::test]
async fn stub_returns_responses_in_order() {
    let client = StubClient::new(vec![
        LlmResponse::text("first"),
        LlmResponse::text("second"),
        LlmResponse::text("third"),
    ]);
    let msgs = vec![Message::user("test")];

    let r1 = client.complete(None, &msgs).await.unwrap();
    let r2 = client.complete(None, &msgs).await.unwrap();
    let r3 = client.complete(None, &msgs).await.unwrap();

    assert_eq!(r1.content.as_deref(), Some("first"));
    assert_eq!(r2.content.as_deref(), Some("second"));
    assert_eq!(r3.content.as_deref(), Some("third"));
}

#[tokio::test]
async fn stub_errors_when_exhausted() {
    let client = StubClient::new(vec![LlmResponse::text("only one")]);
    let msgs = vec![Message::user("test")];

    client.complete(None, &msgs).await.unwrap();
    let err = client.complete(None, &msgs).await;
    assert!(err.is_err());
    assert!(err.unwrap_err().to_string().contains("exhausted"));
}

#[test]
fn is_final_answer_when_no_tool_call() {
    assert!(LlmResponse::text("done").is_final_answer());
}

#[test]
fn not_final_when_tool_call_present() {
    let r = LlmResponse::tool("calculator", serde_json::json!({"expression": "1+1"}));
    assert!(!r.is_final_answer());
}

#[test]
fn message_system_sets_role() {
    let m = Message::system("you are an agent");
    assert_eq!(m.role, "system");
    assert_eq!(m.content, "you are an agent");
}

#[test]
fn message_user_sets_role() {
    let m = Message::user("hello");
    assert_eq!(m.role, "user");
    assert_eq!(m.content, "hello");
}

#[test]
fn message_assistant_sets_role() {
    let m = Message::assistant("hi there");
    assert_eq!(m.role, "assistant");
    assert_eq!(m.content, "hi there");
}

#[test]
fn llm_response_text_has_content_no_tool_call() {
    let r = LlmResponse::text("answer");
    assert_eq!(r.content.as_deref(), Some("answer"));
    assert!(r.tool_call.is_none());
}

#[test]
fn llm_response_tool_has_tool_call_no_content() {
    let args = serde_json::json!({"x": 1});
    let r = LlmResponse::tool("my_tool", args.clone());
    assert!(r.content.is_none());
    let tc = r.tool_call.unwrap();
    assert_eq!(tc.name, "my_tool");
    assert_eq!(tc.arguments, args);
}

#[test]
fn llm_response_neither_is_not_final() {
    let r = LlmResponse {
        content: None,
        tool_call: None,
    };
    assert!(!r.is_final_answer());
}

#[test]
fn tool_call_equality() {
    let a = ToolCall {
        name: "calc".to_string(),
        arguments: serde_json::json!({"expression": "1+1"}),
    };
    let b = ToolCall {
        name: "calc".to_string(),
        arguments: serde_json::json!({"expression": "1+1"}),
    };
    assert_eq!(a, b);
}

#[tokio::test]
async fn box_dyn_client_delegates() {
    let inner = StubClient::new(vec![LlmResponse::text("delegated")]);
    let boxed: Box<dyn LlmClient> = Box::new(inner);
    let result = boxed
        .complete(None, &[Message::user("test")])
        .await
        .unwrap();
    assert_eq!(result.content.as_deref(), Some("delegated"));
}

#[test]
fn message_serde_round_trip() {
    let m = Message::system("be helpful");
    let json = serde_json::to_string(&m).unwrap();
    let m2: Message = serde_json::from_str(&json).unwrap();
    assert_eq!(m2.role, "system");
    assert_eq!(m2.content, "be helpful");
}

#[test]
fn llm_response_text_serde_round_trip() {
    let r = LlmResponse::text("the answer");
    let json = serde_json::to_string(&r).unwrap();
    let r2: LlmResponse = serde_json::from_str(&json).unwrap();
    assert_eq!(r2.content.as_deref(), Some("the answer"));
    assert!(r2.tool_call.is_none());
}

#[test]
fn llm_response_tool_serde_round_trip() {
    let r = LlmResponse::tool("calc", serde_json::json!({"expr": "1+1"}));
    let json = serde_json::to_string(&r).unwrap();
    let r2: LlmResponse = serde_json::from_str(&json).unwrap();
    let tc = r2.tool_call.unwrap();
    assert_eq!(tc.name, "calc");
}

#[test]
fn tool_call_serde_round_trip() {
    let tc = ToolCall {
        name: "my_tool".to_string(),
        arguments: serde_json::json!({"x": 42}),
    };
    let json = serde_json::to_string(&tc).unwrap();
    let tc2: ToolCall = serde_json::from_str(&json).unwrap();
    assert_eq!(tc2.name, "my_tool");
    assert_eq!(tc2.arguments["x"], 42);
}

#[test]
fn claude_client_new_empty_model_uses_default() {
    use yaai_llm::AnthropicClient;
    // Empty model string should fall back to the built-in default — just verify construction succeeds
    let _ = AnthropicClient::new("test-api-key", "");
}

#[test]
fn claude_client_new_explicit_model() {
    use yaai_llm::AnthropicClient;
    let _ = AnthropicClient::new("test-api-key", "claude-3-opus");
}

#[test]
fn openai_client_new() {
    use yaai_llm::OpenAiClient;
    let _ = OpenAiClient::new("test-api-key", "gpt-4o");
}
