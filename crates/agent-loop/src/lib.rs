//! Agent execution loop — the ReAct pattern (observe → think → act).
//!
//! [`AgentRunner`] drives one agent through its loop until the LLM produces a
//! final answer or `max_steps` is reached.

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use tracing::{info, warn};
use uuid::Uuid;
use yaai_llm::{LlmClient, Message};
use yaai_memory::{Role, SessionMemory};
use yaai_tools::ToolRegistry;
use yaai_tracer::{EventKind, Tracer};

/// Configuration for a single agent instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    /// Unique identifier (used in traces and logs).
    pub id: String,
    /// System prompt that frames the agent's role and available tools.
    pub system_prompt: String,
    /// Maximum loop iterations before the run is aborted with an error.
    pub max_steps: u32,
}

/// The outcome of a completed agent run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentResult {
    pub run_id: Uuid,
    pub agent_id: String,
    /// The final text answer produced by the agent.
    pub answer: String,
    /// Number of loop iterations consumed.
    pub steps_taken: u32,
}

/// Drives an agent through its ReAct execution loop.
pub struct AgentRunner<'a> {
    config: &'a AgentConfig,
    llm: &'a dyn LlmClient,
    tools: &'a ToolRegistry,
    tracer: &'a Tracer,
    memory: &'a mut SessionMemory,
}

impl<'a> AgentRunner<'a> {
    pub fn new(
        config: &'a AgentConfig,
        llm: &'a dyn LlmClient,
        tools: &'a ToolRegistry,
        tracer: &'a Tracer,
        memory: &'a mut SessionMemory,
    ) -> Self {
        Self {
            config,
            llm,
            tools,
            tracer,
            memory,
        }
    }

    /// Run the agent loop on the given task, returning the final answer.
    ///
    /// The caller is responsible for calling [`Tracer::close`] on the tracer
    /// after this method returns (success or error) to shut down the background
    /// writer task cleanly.
    pub async fn run(&mut self, task: impl Into<String>) -> Result<AgentResult> {
        let task = task.into();
        let run_id = self.tracer.run_id();

        info!(agent = %self.config.id, %run_id, %task, "agent starting");

        self.memory.add(Role::User, &task);

        for step in 0..self.config.max_steps {
            let messages: Vec<Message> = self
                .memory
                .entries()
                .iter()
                .map(|e| Message {
                    // Role::Tool has no direct equivalent in the current Message format
                    // (which lacks tool_call_id), so tool observations are surfaced to
                    // the LLM as user messages to maintain API compatibility.
                    role: match &e.role {
                        Role::Tool => "user".to_string(),
                        other => other.to_string(),
                    },
                    content: e.content.clone(),
                })
                .collect();

            self.tracer.emit(
                &self.config.id,
                step,
                EventKind::Prompt,
                serde_json::json!({ "message_count": messages.len() }),
            )?;

            let response = self
                .llm
                .complete(Some(&self.config.system_prompt), &messages)
                .await?;

            if response.content.is_none() && response.tool_call.is_none() {
                let msg = format!(
                    "agent '{}' received an empty LLM response at step {}",
                    self.config.id, step
                );
                self.tracer.emit(
                    &self.config.id,
                    step,
                    EventKind::Error,
                    serde_json::json!({ "error": &msg }),
                )?;
                self.tracer.flush().await?;
                bail!(msg);
            }

            if let Some(ref text) = response.content {
                self.tracer
                    .emit(&self.config.id, step, EventKind::Decision, text)?;
                self.memory.add(Role::Assistant, text);
            }

            if let Some(tc) = &response.tool_call {
                info!(agent = %self.config.id, tool = %tc.name, step, "tool call");

                self.tracer.emit(
                    &self.config.id,
                    step,
                    EventKind::ToolCall,
                    serde_json::json!({ "tool": tc.name, "args": tc.arguments }),
                )?;

                let observation = match self.tools.dispatch(&tc.name, tc.arguments.clone()).await {
                    Ok(result) => {
                        self.tracer
                            .emit(&self.config.id, step, EventKind::ToolResult, &result)?;
                        result.to_string()
                    }
                    Err(e) => {
                        let msg = format!("Tool error: {e}");
                        self.tracer.emit(
                            &self.config.id,
                            step,
                            EventKind::Error,
                            serde_json::json!({ "error": &msg }),
                        )?;
                        warn!(agent = %self.config.id, error = %e, "tool execution failed");
                        msg
                    }
                };

                self.memory.add(
                    Role::Tool,
                    format!("Tool '{}' returned: {}", tc.name, observation),
                );
            } else if response.is_final_answer() {
                let answer = response.content.unwrap_or_default();
                info!(agent = %self.config.id, step, "final answer");

                self.tracer
                    .emit(&self.config.id, step, EventKind::FinalAnswer, &answer)?;

                self.tracer.flush().await?;

                return Ok(AgentResult {
                    run_id,
                    agent_id: self.config.id.clone(),
                    answer,
                    steps_taken: step + 1,
                });
            }
        }

        warn!(
            agent = %self.config.id,
            max = self.config.max_steps,
            "max steps reached without final answer"
        );
        self.tracer.flush().await?;
        bail!(
            "agent '{}' reached max_steps ({}) without a final answer",
            self.config.id,
            self.config.max_steps
        );
    }
}
