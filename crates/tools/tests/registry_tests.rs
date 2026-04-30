use std::io::Write;
use tempfile::NamedTempFile;
use yaai_tools::{
    builtin::GrepFilesTool, builtin::ReadTool, ToolError, ToolRegistry, ToolSchemaFormat,
};

// --- ToolError display contract tests ---

#[test]
fn tool_error_not_found_display() {
    let err = ToolError::NotFound("my_tool".to_string());
    assert!(err.to_string().contains("my_tool"));
    assert!(err.to_string().contains("not found"));
}

#[test]
fn tool_error_invalid_input_display() {
    let err = ToolError::InvalidInput {
        name: "read".to_string(),
        reason: "bad field".to_string(),
    };
    assert!(err.to_string().contains("read"));
    assert!(err.to_string().contains("bad field"));
}

#[test]
fn tool_error_execution_failed_display() {
    let err = ToolError::ExecutionFailed {
        name: "read".to_string(),
        reason: "timeout".to_string(),
    };
    assert!(err.to_string().contains("read"));
    assert!(err.to_string().contains("timeout"));
}

// --- Registry contract tests (tool-agnostic) ---

#[test]
fn lists_registered_tools() {
    let r = ToolRegistry::new().register(ReadTool::new());
    assert!(r.names().contains(&"read"));
}

#[tokio::test]
async fn dispatch_unknown_tool_returns_not_found() {
    let registry = ToolRegistry::new();
    let err = registry
        .dispatch("does_not_exist", serde_json::json!({}))
        .await
        .unwrap_err();
    assert!(err.to_string().contains("not found"));
}

#[tokio::test]
async fn dispatch_missing_required_field_returns_invalid_input() {
    let registry = ToolRegistry::new().register(ReadTool::new());

    match registry.dispatch("read", serde_json::json!({})).await {
        Err(ToolError::InvalidInput { name, reason }) => {
            assert_eq!(name, "read");
            assert!(reason.contains("missing required field 'file_path'"));
        }
        _ => panic!("expected InvalidInput error"),
    }
}

#[tokio::test]
async fn dispatch_null_required_field_returns_invalid_input() {
    let registry = ToolRegistry::new().register(ReadTool::new());

    match registry
        .dispatch("read", serde_json::json!({ "file_path": null }))
        .await
    {
        Err(ToolError::InvalidInput { reason, .. }) => {
            assert!(reason.contains("missing required field 'file_path'"));
        }
        _ => panic!("expected InvalidInput error for null required field"),
    }
}

#[test]
fn descriptions_anthropic_uses_input_schema_key() {
    let registry = ToolRegistry::new().register(ReadTool::new());

    let descs = registry.descriptions(ToolSchemaFormat::Anthropic);
    assert_eq!(descs.len(), 1);
    assert!(descs[0].get("name").is_some());
    assert!(descs[0].get("description").is_some());
    assert!(descs[0].get("input_schema").is_some());
    assert!(descs[0].get("parameters").is_none());
    assert!(descs[0].get("function").is_none());
}

#[test]
fn descriptions_openai_uses_function_parameters_key() {
    let registry = ToolRegistry::new().register(ReadTool::new());

    let descs = registry.descriptions(ToolSchemaFormat::OpenAi);
    assert_eq!(descs.len(), 1);
    assert_eq!(descs[0]["type"], "function");
    let func = &descs[0]["function"];
    assert!(func.get("parameters").is_some());
    assert!(func.get("input_schema").is_none());
    assert_eq!(func["name"], "read");
}

// --- GrepFiles tool integration via registry ---

#[test]
fn grep_files_tool_appears_in_registry() {
    let dir = tempfile::tempdir().unwrap();
    let registry = ToolRegistry::new().register(GrepFilesTool::with_working_dir(dir.path()));
    assert!(registry.names().contains(&"grep_files"));
}

#[tokio::test]
async fn dispatch_grep_files_missing_pattern_returns_invalid_input() {
    let dir = tempfile::tempdir().unwrap();
    let registry = ToolRegistry::new().register(GrepFilesTool::with_working_dir(dir.path()));

    match registry.dispatch("grep_files", serde_json::json!({})).await {
        Err(ToolError::InvalidInput { name, reason }) => {
            assert_eq!(name, "grep_files");
            assert!(reason.contains("missing required field 'pattern'"));
        }
        _ => panic!("expected InvalidInput for missing pattern"),
    }
}

#[tokio::test]
async fn dispatch_grep_files_returns_matching_file() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("lib.rs"), "pub fn grep_target() {}").unwrap();
    std::fs::write(dir.path().join("other.rs"), "pub fn unrelated() {}").unwrap();

    let registry = ToolRegistry::new().register(GrepFilesTool::with_working_dir(dir.path()));

    let result = registry
        .dispatch(
            "grep_files",
            serde_json::json!({ "pattern": "grep_target" }),
        )
        .await
        .unwrap();

    let files = result["files"].as_array().unwrap();
    assert_eq!(files.len(), 1);
    assert!(files[0].as_str().unwrap().contains("lib.rs"));
    assert_eq!(result["truncated"], false);
}

#[tokio::test]
async fn dispatch_grep_files_no_matches_returns_empty_ok() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("a.rs"), "fn main() {}").unwrap();

    let registry = ToolRegistry::new().register(GrepFilesTool::with_working_dir(dir.path()));

    let result = registry
        .dispatch(
            "grep_files",
            serde_json::json!({ "pattern": "this_pattern_does_not_exist_xyz" }),
        )
        .await
        .unwrap();

    assert_eq!(result["files"].as_array().unwrap().len(), 0);
    assert_eq!(result["total"], 0);
    assert_eq!(result["truncated"], false);
}

// --- Read tool integration via registry ---

#[tokio::test]
async fn dispatch_read_returns_file_contents() {
    let mut f = NamedTempFile::new().unwrap();
    writeln!(f, "line one").unwrap();
    writeln!(f, "line two").unwrap();

    let registry = ToolRegistry::new().register(ReadTool::new());

    let result = registry
        .dispatch(
            "read",
            serde_json::json!({ "file_path": f.path().to_str().unwrap() }),
        )
        .await
        .unwrap();

    assert_eq!(result["type"], "file");
    assert_eq!(result["lines"]["total"], 2);
    let content = result["content"].as_str().unwrap();
    assert!(content.contains("1: line one"));
    assert!(content.contains("2: line two"));
}

#[tokio::test]
async fn dispatch_read_missing_file_returns_execution_failed() {
    let registry = ToolRegistry::new().register(ReadTool::new());

    match registry
        .dispatch(
            "read",
            serde_json::json!({ "file_path": "/nonexistent/path/file.txt" }),
        )
        .await
    {
        Err(ToolError::ExecutionFailed { name, .. }) => assert_eq!(name, "read"),
        _ => panic!("expected ExecutionFailed"),
    }
}
