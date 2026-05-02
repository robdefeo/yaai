//! Agent execution loop — the ReAct pattern (observe → think → act).
//!
//! [`AgentRunner`] drives one agent through its loop until the LLM produces a
//! final answer or `max_steps` is reached.

use anyhow::{bail, Result};
use futures::future;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tracing::{info, warn};
use uuid::Uuid;
use yaai_llm::{ConversationTurn, LlmClient, Message, ToolCall};
use yaai_memory::{EntryContent, MemoryToolCall, Role, SessionMemory};
use yaai_tools::{ToolRegistry, ToolSchemaFormat};
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
    /// Final conversation history. Skipped during serialisation to keep
    /// the JSON output lean; inspect it directly when testing or introspecting.
    #[serde(skip)]
    pub memory: SessionMemory,
}

/// Drives an agent through its ReAct execution loop.
pub struct AgentRunner<'a> {
    config: &'a AgentConfig,
    llm: &'a dyn LlmClient,
    tools: &'a ToolRegistry,
    tracer: &'a Tracer,
    memory: SessionMemory,
    tool_format: ToolSchemaFormat,
    token_tx: Option<mpsc::UnboundedSender<String>>,
}

impl<'a> AgentRunner<'a> {
    pub fn new(
        config: &'a AgentConfig,
        llm: &'a dyn LlmClient,
        tools: &'a ToolRegistry,
        tracer: &'a Tracer,
        tool_format: ToolSchemaFormat,
    ) -> Self {
        Self {
            config,
            llm,
            tools,
            tracer,
            memory: SessionMemory::new(),
            tool_format,
            token_tx: None,
        }
    }

    /// Enable token streaming. Text deltas are sent to `tx` as they arrive
    /// from the LLM; the full [`LlmResponse`] is still returned as normal.
    pub fn with_streaming(self, tx: mpsc::UnboundedSender<String>) -> Self {
        Self {
            token_tx: Some(tx),
            ..self
        }
    }

    /// Seed the runner with existing conversation history.
    ///
    /// Use this for multi-turn conversations where the history from a previous
    /// run should be carried forward. For a fresh, stateless run, omit this call.
    pub fn with_memory(self, memory: SessionMemory) -> Self {
        Self { memory, ..self }
    }

    /// Run the agent loop on the given task, returning the final answer.
    ///
    /// Consumes the runner. The returned [`AgentResult`] includes the final
    /// conversation history so callers can inspect it without holding a separate
    /// mutable reference throughout the run.
    ///
    /// The caller is responsible for calling [`Tracer::close`] on the tracer
    /// after this method returns (success or error) to shut down the background
    /// writer task cleanly.
    pub async fn run(mut self, task: impl Into<String>) -> Result<AgentResult> {
        let task = task.into();
        let run_id = self.tracer.run_id();

        info!(agent = %self.config.id, %run_id, %task, "agent starting");

        self.memory.add(Role::User, &task);

        let tool_descriptors = self.tools.descriptions(self.tool_format);

        for step in 0..self.config.max_steps {
            // Build structured conversation turns from memory for the API call.
            let turns: Vec<ConversationTurn> = self
                .memory
                .entries()
                .iter()
                .map(|e| match &e.content {
                    EntryContent::Text { text } => ConversationTurn::Text(Message {
                        role: e.role.to_string(),
                        content: text.clone(),
                    }),
                    EntryContent::ToolCall { calls, reasoning } => {
                        ConversationTurn::AssistantToolCall {
                            calls: calls
                                .iter()
                                .map(|c| ToolCall {
                                    id: c.id.clone(),
                                    name: c.name.clone(),
                                    arguments: c.arguments.clone(),
                                })
                                .collect(),
                            reasoning: reasoning.clone(),
                        }
                    }
                    EntryContent::ToolResult {
                        tool_call_id,
                        content,
                    } => ConversationTurn::ToolResult {
                        tool_call_id: tool_call_id.clone(),
                        content: content.clone(),
                    },
                })
                .collect();

            self.tracer.emit(
                &self.config.id,
                step,
                EventKind::Prompt,
                serde_json::json!({ "turn_count": turns.len() }),
            )?;

            let response = match self.token_tx.as_ref() {
                Some(tx) => {
                    self.llm
                        .complete_streaming(
                            Some(&self.config.system_prompt),
                            &turns,
                            &tool_descriptors,
                            tx,
                        )
                        .await?
                }
                None => {
                    self.llm
                        .complete(Some(&self.config.system_prompt), &turns, &tool_descriptors)
                        .await?
                }
            };

            if response.content.is_none() && response.tool_calls.is_empty() {
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
                // Emit a Decision trace for reasoning text regardless of whether a
                // tool call follows — reasoning is always worth tracing. When a tool
                // call is present the text is *not* added to memory here; instead it
                // is stored inside the ToolCall entry below so both are replayed as a
                // single assistant message (required by Anthropic and OpenAI).
                self.tracer
                    .emit(&self.config.id, step, EventKind::Decision, text)?;
                // Only store a standalone text entry when there is no tool call.
                // When a tool call is present, the reasoning is stored inside it.
                if response.tool_calls.is_empty() {
                    self.memory.add(Role::Assistant, text);
                }
            }

            if !response.tool_calls.is_empty() {
                for tc in &response.tool_calls {
                    info!(agent = %self.config.id, tool = %tc.name, step, "tool call");
                    self.tracer.emit(
                        &self.config.id,
                        step,
                        EventKind::ToolCall,
                        serde_json::json!({ "tool": tc.name, "args": tc.arguments }),
                    )?;
                }

                // Store reasoning alongside all tool calls so the batch is replayed
                // as a single assistant message — required by Anthropic and OpenAI.
                self.memory.add_entry(
                    Role::Assistant,
                    EntryContent::ToolCall {
                        calls: response
                            .tool_calls
                            .iter()
                            .map(|tc| MemoryToolCall {
                                id: tc.id.clone(),
                                name: tc.name.clone(),
                                arguments: tc.arguments.clone(),
                            })
                            .collect(),
                        reasoning: response.content.clone(),
                    },
                );

                // Dispatch all tool calls concurrently.
                let tools = self.tools;
                let dispatch_futs: Vec<_> = response
                    .tool_calls
                    .iter()
                    .map(|tc| {
                        let name = tc.name.clone();
                        let args = tc.arguments.clone();
                        async move { tools.dispatch(&name, args).await }
                    })
                    .collect();
                let dispatch_results = future::join_all(dispatch_futs).await;

                for (tc, result) in response.tool_calls.iter().zip(dispatch_results) {
                    let observation = match result {
                        Ok(val) => {
                            self.tracer
                                .emit(&self.config.id, step, EventKind::ToolResult, &val)?;
                            val.to_string()
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

                    self.memory.add_entry(
                        Role::User,
                        EntryContent::ToolResult {
                            tool_call_id: tc.id.clone(),
                            content: observation,
                        },
                    );
                }
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
                    memory: self.memory,
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
