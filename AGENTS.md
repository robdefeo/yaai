# AGENTS.md — yaai Agent Harness

Guidelines for AI coding agents working in this repository.

## Guidelines

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

**These guidelines are working if: fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.**

## Principles

Design, engineering, leadership, and communication principles for different professional contexts.

---

### Design & Product Management

- **Principle of Least Astonishment (POLA)**: A system component should behave as users expect, not surprising them. Design interfaces and workflows to match mental models and established conventions. Use status indicators, clear feedback, visible affordances.
- **Visibility**: System state should be immediately observable. Users shouldn't guess what's happening. Provide status indicators, clear feedback, and visible affordances.
- **Progressive Disclosure**: Reveal complexity only when needed. Show basics by default, hide advanced options until relevant. Scaffold learning progressively.
- **Consistency**: Apply the same rules and patterns across similar situations. Use uniform naming, replicate patterns across interfaces, document deviations explicitly.

### Engineering & Development

- **Keep It Simple, Stupid (KISS)**: Simpler solutions are preferable to complex ones. Resist over-engineering. Choose straightforward approaches. Minimize moving parts. Document why complexity was necessary when it is.
- **Don't Repeat Yourself (DRY)**: Eliminate redundancy; maintain a single source of truth. Duplication creates divergence. Extract shared logic, use libraries, refactor repeated patterns.
- **Separation of Concerns**: Each module should have a single, well-defined responsibility. Single-responsibility modules are predictable. Design clear interfaces, isolate concerns, test independently.
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

## Build Commands

```bash
just build              # build all crates
just test               # cargo test + bun test
just lint               # cargo fmt --check + cargo clippy -- -D warnings + biome check
just fmt                # cargo fmt + biome format --write
just coverage           # cargo-llvm-cov coverage check
cargo run -p yaai -- -p "your multi word question"
```

## Code Style

- Group by responsibility, not type
- Avoid large modules, split early

### Rust

- Architect to prefer immutability: avoid `&mut self` and `let mut` where a value can be constructed or transformed instead; use interior mutability (`AtomicT`, `Mutex`) only when the trait or concurrency boundary requires it.
- No `unwrap()` in library code — use `?` and `anyhow`
- Tests in `tests/` (integration) or inline `#[cfg(test)]` (unit)
- Prefer `tests/` for integration tests; add `benches/` or `examples/` only when they add value.
- `lib.rs` defines public API and high-level module structure, not implementation details.


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
- Test executable production lines. Coverage is measured with `cargo-llvm-cov`; there is no comment-based line exclusion on stable Rust. Keep coverage meaningful by testing real production paths.
- Run `just test` before any commit

## Commit and PR Rules

- Commit message format and PR title must follow `.config/commitlint.config.mjs`
