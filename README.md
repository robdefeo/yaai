# yaai — POC Agent Harness

A minimal, extensible agent harness implementing the ReAct pattern
(observe → think → act).

## Architecture

```
User Input
    ↓
Orchestrator  (single or sequential multi-agent)
    ↓
Agent Loop    (agent-loop: ReAct)
    ↓
LLM ──► Tool Registry ──► Memory
    ↓
Tracer        (JSON trace per run)
    ↓
Final Output
```

## Crates

| Crate | Purpose |
|-------|---------|
| `yaai-tracer` | Structured JSON trace emitter |
| `yaai-memory` | Session-scoped context store |
| `yaai-llm` | LLM abstraction — OpenAI client + deterministic stub |
| `yaai-tools` | Tool registry + built-ins (calculator, shell_exec, web_search) |
| `yaai-agent-loop` | ReAct execution loop |
| `yaai-orchestrator` | Single + sequential multi-agent workflows |
| `yaai` | CLI: `yaai run / tools / trace` |

## Prerequisites

- [mise](https://mise.jdx.dev/) — `curl https://mise.run | sh`

```bash
just install   # installs Rust 1.93.1, bun 1.2.5, cargo-watch, grcov, just via mise
just build     # build everything
just test      # cargo test + bun test
just lint      # fmt check + clippy -D warnings + biome
```

## Quickstart

```bash
export OPENAI_API_KEY=sk-...
just run -- run configs/examples/research-assistant.yaml "What is the ReAct pattern?"
```

## Use Cases

| Config | Description |
|--------|-------------|
| `configs/examples/research-assistant.yaml` | UC1: query → search → summarise |
| `configs/examples/coding-agent.yaml` | UC2: write code → execute → debug |
| `configs/examples/multi-agent.yaml` | UC3: planner → executor → reviewer |

## CLI Commands

```bash
# Run a workflow
yaai run <config.yaml> "<task>"

# List registered tools
yaai tools

# Inspect a saved trace
yaai trace view <run_id>
yaai trace view <run_id> --traces-dir ./my-traces
```

## Adding a Tool

1. Create `crates/tools/src/builtin/my_tool.rs` implementing the `Tool` trait
2. Export from `crates/tools/src/builtin/mod.rs` and `src/lib.rs`
3. Register in `apps/cli/src/commands/run.rs` → `build_tool_registry`
4. Add contract tests in `crates/tools/tests/`

Time to add a new tool: **< 30 min**

## Defining an Agent

```yaml
# my-config.yaml
workflow:
  type: single
  steps: []
agents:
  - id: my-agent
    model: gpt-4o
    tools: [calculator]
    max_steps: 10
    system_prompt: "You are a helpful agent..."
```

```bash
just run -- run my-config.yaml "my task"
```

Time to define a new agent: **< 15 min**

## Trace Inspection

Each run writes a JSON trace to `traces/<run_id>.json`:

```bash
yaai trace view <run_id>
```

Events: `prompt`, `tool_call`, `tool_result`, `decision`, `final_answer`, `error`
