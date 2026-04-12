# AGENTS.md — yaai Agent Harness

Guidelines for AI coding agents working in this repository.

## Principles

Design, engineering, leadership, and communication principles for different professional contexts.

---

### Design & Product Management

- **Principle of Least Astonishment (POLA)**: A system component should behave as users expect, not surprising them. Design interfaces and workflows to match mental models and established conventions. Use status indicators, clear feedback, visible affordances.
- **Visibility**: System state should be immediately observable. Users shouldn't guess what's happening. Provide status indicators, clear feedback, and visible affordances.
- **Progressive Disclosure**: Reveal complexity only when needed. Show basics by default, hide advanced options until relevant. Scaffold learning progressively.
- **Consistency**: Apply the same rules and patterns across similar situations. Use uniform naming, replicate patterns across interfaces, document deviations explicitly.

### Engineering & Development

— **Keep It Simple, Stupid (KISS)**: Simpler solutions are preferable to complex ones. Resist over-engineering. Choose straightforward approaches. Minimize moving parts. Document why complexity was necessary when it is.

- **Don't Repeat Yourself (DRY)**: Eliminate redundancy; maintain a single source of truth. Duplication creates divergence. Extract shared logic, use libraries, refactor repeated patterns.
- **Separation of Concerns**: Each module should have a single, well-defined responsibility Single-responsibility modules are predictable. Design clear interfaces, isolate concerns, test independently.
- **Convention over Configuration**: Provide sensible defaults and standard patterns. Define framework defaults, use standard naming, minimize configuration surfaces.

### Leadership & Management

- **Transparency**: Keep decision-making processes and reasoning visible. Teams are less surprised when they understand the 'why.' Share rationale, document trade-offs, explain constraints.
- **Explicit Constraints**: Clearly communicate boundaries and scope upfront. Define scope, communicate resource limits, state decision boundaries, document non-negotiables.

---

### Writing & Communication

- **Simple is Better Than Complex**: Clear, accessible expression outweighs sophisticated or dense writing. Use short sentences and common words. Remove jargon unless necessary. Value clarity over cleverness. Make complexity visible rather than hidden.
- **Information Scent**: Links, headings, and titles should accurately signal what's inside. Readers shouldn't be surprised by content. Use descriptive links, specific headings, preview scope, fulfill promises.
- **Structural Parallelism**: Parallel sentence structure creates pattern recognition. Parallel structures set expectations that are then met. Use consistent lists, matching syntax, parallel emphasis.

---

## Repo Structure

```
yaai/
├── apps/cli/              # CLI binary (`yaai` command)
│   └── src/
│       ├── main.rs
│       ├── config.rs      # YAML config loading + validation
│       └── commands/      # run, tools, trace subcommands
├── crates/
│   ├── tracer/            # Structured JSON trace emitter
│   ├── memory/            # Session-scoped context store
│   ├── llm/               # LLM client abstraction (StubClient, OpenAiClient)
│   ├── tools/             # Tool registry + built-ins (calculator, shell_exec, web_search)
│   ├── agent-loop/        # ReAct execution loop
│   └── orchestrator/      # Single + sequential multi-agent coordination
├── configs/examples/      # YAML workflow configs
└── traces/                # Runtime trace output (gitignored)
```

## Tool Versions

| Tool  | Version |
|-------|---------|
| Rust  | 1.93.1  |
| Bun   | 1.2.5   |
| just  | 1.38.0  |
| grcov | 0.10.7  |

## Build Commands

```bash
just build              # build all crates
just test               # cargo test + bun test
just lint               # cargo fmt --check + cargo clippy -- -D warnings + biome check
just fmt                # cargo fmt + biome format --write
just coverage-html      # grcov coverage view
just coverage-check     # grcov coverage check
cargo run -p yaai -- -p "your multi word question"
```

## Code Style
- No `unwrap()` in library code — use `?` and `anyhow`
- Tests in `tests/` (integration) or inline `#[cfg(test)]` (unit)
- Every crate has: `src/`, `tests/`, `benches/`, `examples/`
- `lib.rs` defines public API and high-level module structure, not implementation details.
- Group by responsibility, not type
- Avoid giant modules, split early

### Rust
rustfmt (100-char max width) + clippy `-D warnings` — all warnings are errors

- **TypeScript**: Biome (100-char line width, organize imports enabled)


## Crate Responsibilities

| Crate | Responsibility |
|-------|----------------|
| `yaai-tracer` | Event emission + JSON file writing only |
| `yaai-memory` | Session context list, no LLM calls |
| `yaai-llm` | LLM I/O only, no tool dispatch |
| `yaai-tools` | Tool execution only, no agent state |
| `yaai-agent-loop` | ReAct loop — composes llm + tools + memory + tracer |
| `yaai-orchestrator` | Workflow coordination — composes agent-loop instances |
| `yaai` | User-facing CLI — wires everything from config |

## Testing

- Use `StubClient` from `yaai-llm` for all agent tests — **no real LLM calls in tests**
- Assert behavioural invariants: termination, trace event sequence, memory growth
- Run `just test` before any commit

## Other

- Commit message format and PR title must follow `.config/commitlint.config.mjs`
- No Co-authored from commit messages or PR's