# Harold

An AI improviser and comedy coach built with [Pydantic AI](https://ai.pydantic.dev/) and [BAML](https://docs.boundaryml.com/). Harold is your scene partner for practicing long-form improv comedy, grounded in [UCB Comedy Improvisation Manual](https://www.amazon.com/Upright-Citizens-Brigade-Comedy-Improvisation/dp/0989387801) principles.

## Quick Start

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
# Clone and install
git clone <repo-url> && cd harold
uv sync

# Configure your API key
export ANTHROPIC_API_KEY=sk-ant-...

# Start improvising
uv run harold
```

Harold will greet you and you can begin a scene by saying something in character. Type `quit`, `exit`, or `q` to end the session.

## Architecture

```
User (CLI / API)
    |
    v
Interfaces (cli.py / api.py)
    |
    v
Scene Partner Agent  (Pydantic AI Agent[HaroldDependencies, SceneResponse])
    |
    +-- Short-term memory: Pydantic AI message_history
    +-- Long-term memory: LongTermMemory protocol (in-memory / pgvector)
    +-- Trajectory memory: TrajectoryMemory protocol (in-memory / Neo4j)
    +-- Observability: OpenTelemetry -> Arize Phoenix
```

- **Pydantic AI** handles agent orchestration, tool calling, dependency injection, and conversation history.
- **BAML** manages prompt templates and standalone structured extractions (scene summarization, knowledge extraction).
- **Memory backends** are swappable via Protocol-based abstractions. The MVP uses in-memory stores; pgvector and Neo4j backends are planned.
- **Interfaces** are decoupled from core logic. CLI ships first; a FastAPI WebSocket interface is planned.

## Project Structure

```
src/harold/
    __main__.py          # Entry point (python -m harold)
    config.py            # Settings via pydantic-settings (HAROLD_* env vars)
    dependencies.py      # Dependency container for agent injection
    observability.py     # OpenTelemetry + Phoenix bootstrap
    agents/
        scene_partner.py # Core improv agent definition
        prompts.py       # UCB-grounded system prompt
    tools/
        memory_tools.py  # Scene recall and knowledge lookup
        scene_tools.py   # Scene start/end lifecycle
    memory/
        base.py          # LongTermMemory + TrajectoryMemory protocols
        backends/
            in_memory.py # MVP keyword-matching backends
    models/
        scene.py         # SceneResponse, SceneState, Turn, SceneSummary
        memory.py        # KnowledgeEntry, KnowledgeCategory
        types.py         # Shared annotated types and constants
    interfaces/
        cli.py           # Rich-based terminal REPL
baml_src/
    clients.baml         # LLM provider configuration
    scene.baml           # Scene analysis types and functions
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | (required) | Anthropic API key for Claude access |
| `HAROLD_LLM_MODEL` | `anthropic:claude-sonnet-4-20250514` | Pydantic AI model identifier |
| `HAROLD_LLM_TEMPERATURE` | `0.85` | Sampling temperature (0.0-2.0) |
| `HAROLD_MEMORY_BACKEND` | `in_memory` | Long-term memory backend |
| `HAROLD_TRAJECTORY_BACKEND` | `in_memory` | Trajectory memory backend |
| `HAROLD_PHOENIX_ENABLED` | `false` | Enable Arize Phoenix tracing |
| `HAROLD_PHOENIX_ENDPOINT` | `http://127.0.0.1:6006/v1/traces` | OTLP endpoint for Phoenix |

Copy `.env.example` to `.env` and fill in your API key to get started.

## Observability

Harold integrates with [Arize Phoenix](https://docs.arize.com/phoenix) for observability. When enabled, every agent run, LLM call, tool invocation, and token usage is traced via OpenTelemetry.

```bash
# Install the Phoenix UI (one-time)
uv sync --extra phoenix

# Start Phoenix in one terminal
phoenix serve

# Run Harold with tracing in another terminal
HAROLD_PHOENIX_ENABLED=true uv run harold
```

Open [http://localhost:6006](http://localhost:6006) to see traces. Each agent turn produces a trace showing the LLM request/response, any tool calls, token counts, and latency breakdown.

## pgvector Memory Backend

For semantic search over past scenes and improv knowledge, Harold supports PostgreSQL with pgvector.

```bash
# Start PostgreSQL with pgvector
docker compose up -d

# Set environment variables
export OPENAI_API_KEY=sk-...
export HAROLD_MEMORY_BACKEND=pgvector
export HAROLD_PG_DSN=postgresql://postgres:postgres@localhost/harold

# Seed the knowledge base with UCB improv principles (one-time)
uv run python scripts/seed_knowledge.py

# Run Harold with semantic memory
uv run harold
```

When using pgvector, the agent embeds scene summaries and knowledge entries via OpenAI's `text-embedding-3-small` model and stores them alongside their vectors. Searches use cosine similarity instead of keyword matching.

## Development

```bash
# Install with dev dependencies
uv sync --extra dev

# Run the full test suite
uv run pytest

# Run a specific test file
uv run pytest tests/test_memory.py -v

# Lint and type check
uv run ruff check src/ tests/
uv run mypy src/

# Regenerate BAML client after editing baml_src/
baml generate
```

Tests use `StubModel` (aliased from Pydantic AI's `TestModel`) to avoid real API calls. The test suite includes parametrized tests and property-based tests via Hypothesis.
