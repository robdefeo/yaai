//! Tool registry and built-in tools.
//!
//! Register tools with a name, JSON Schema input description, and async execute
//! function. The registry validates required fields before dispatching.

pub mod builtin;

use async_trait::async_trait;
use serde_json::Value;
use thiserror::Error;

pub use builtin::ReadTool;

/// Errors that can occur during tool execution.
#[derive(Debug, Error)]
pub enum ToolError {
    #[error("tool '{0}' not found")]
    NotFound(String),

    #[error("invalid input for tool '{name}': {reason}")]
    InvalidInput { name: String, reason: String },

    #[error("tool '{name}' execution failed: {reason}")]
    ExecutionFailed { name: String, reason: String },
}

/// A tool that can be called by an agent.
#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    /// JSON Schema object describing the expected input.
    fn input_schema(&self) -> Value;
    async fn execute(&self, input: Value) -> Result<Value, ToolError>;
}

/// The LLM provider, used to format tool descriptors correctly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolSchemaFormat {
    /// Anthropic Messages API — uses `input_schema`.
    Anthropic,
    /// OpenAI Chat Completions API — uses `parameters` nested under `function`.
    OpenAi,
}

/// Registry of available tools.
#[derive(Default)]
pub struct ToolRegistry {
    tools: Vec<Box<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a tool, returning `self` to allow chaining.
    pub fn register(mut self, tool: impl Tool + 'static) -> Self {
        self.tools.push(Box::new(tool));
        self
    }

    /// Names of all registered tools.
    pub fn names(&self) -> Vec<&str> {
        self.tools.iter().map(|t| t.name()).collect()
    }

    /// Tool descriptors formatted for the given provider.
    ///
    /// - Anthropic: `{ name, description, input_schema }`
    /// - OpenAI: `{ type: "function", function: { name, description, parameters } }`
    pub fn descriptions(&self, format: ToolSchemaFormat) -> Vec<Value> {
        self.tools
            .iter()
            .map(|t| match format {
                ToolSchemaFormat::Anthropic => serde_json::json!({
                    "name": t.name(),
                    "description": t.description(),
                    "input_schema": t.input_schema(),
                }),
                ToolSchemaFormat::OpenAi => serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": t.name(),
                        "description": t.description(),
                        "parameters": t.input_schema(),
                    }
                }),
            })
            .collect()
    }

    /// Dispatch a tool call by name, validating required fields first.
    pub async fn dispatch(&self, name: &str, input: Value) -> Result<Value, ToolError> {
        let tool = self
            .tools
            .iter()
            .find(|t| t.name() == name)
            .ok_or_else(|| ToolError::NotFound(name.to_string()))?;

        validate_required(tool.as_ref(), &input)?;

        tracing::debug!(tool = name, "dispatching tool");
        tool.execute(input).await
    }
}

/// Validates that all required fields are present and non-null.
fn validate_required(tool: &dyn Tool, input: &Value) -> Result<(), ToolError> {
    let schema = tool.input_schema();
    if let Some(required) = schema.get("required").and_then(|r| r.as_array()) {
        for field in required {
            if let Some(field_name) = field.as_str() {
                match input.get(field_name) {
                    None | Some(Value::Null) => {
                        return Err(ToolError::InvalidInput {
                            name: tool.name().to_string(),
                            reason: format!("missing required field '{field_name}'"),
                        });
                    }
                    _ => {}
                }
            }
        }
    }
    Ok(())
}
