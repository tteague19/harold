# Harold

An AI improviser and comedy coach built with [Pydantic AI](https://ai.pydantic.dev/) and [BAML](https://docs.boundaryml.com/). Harold is your scene partner for practicing long-form improv comedy, grounded in [UCB Comedy Improvisation Manual](https://www.amazon.com/Upright-Citizens-Brigade-Comedy-Improvisation/dp/0989387801) principles.

## Quick Start

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
# Clone and install
git clone <repo-url> && cd harold
uv sync

# Configure your API key
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY

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
- **Memory backends** are swappable via Protocol-based abstractions. The default uses in-memory stores; pgvector provides semantic search for long-term memory, and Neo4j provides graph-based trajectory tracking.
- **Interfaces** are decoupled from core logic. Both a Rich CLI and a FastAPI WebSocket API are available, sharing the same agent and dependency wiring via `bootstrap.py`.

## Project Structure

```
src/harold/
    __main__.py          # Entry point (python -m harold)
    config.py            # Settings via pydantic-settings (HAROLD_* env vars)
    bootstrap.py         # Shared dependency wiring for CLI and API
    dependencies.py      # Dependency container for agent injection
    observability.py     # OpenTelemetry + Phoenix bootstrap
    agents/
        scene_partner.py # Core improv agent definition
        coach.py         # On-demand coaching agent
        prompts.py       # UCB-grounded system prompt
    tools/
        memory_tools.py  # Scene recall and knowledge lookup
        scene_tools.py   # Scene start/end lifecycle
        coaching_tools.py # Trajectory query tools for coach
    memory/
        base.py          # LongTermMemory + TrajectoryMemory protocols
        embeddings.py    # Pydantic AI Embedder wrapper with caching
        backends/
            in_memory.py # MVP keyword-matching backends
            pgvector.py  # PostgreSQL + pgvector semantic search
            neo4j.py     # Neo4j graph trajectory backend
    models/
        scene.py         # SceneResponse, SceneState, Turn, SceneSummary
        memory.py        # KnowledgeEntry, KnowledgeCategory
        coaching.py      # CoachingFeedback structured output
        techniques.py    # Shared CORE_TECHNIQUES reference list
        types.py         # Shared annotated types and constants
    interfaces/
        cli.py           # Rich-based terminal REPL
        api.py           # FastAPI WebSocket server
baml_src/
    clients.baml         # LLM provider configuration
    scene.baml           # Scene analysis types and functions
scripts/
    seed_knowledge.py    # Populate pgvector with UCB improv principles
```

## Environment Variables

All `HAROLD_*` variables are loaded from a `.env` file automatically via pydantic-settings. Copy `.env.example` to `.env` and fill in your keys to get started.

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | (required) | Anthropic API key for Claude access |
| `OPENAI_API_KEY` | (required for pgvector) | OpenAI API key for embeddings |
| `HAROLD_LLM_MODEL` | `anthropic:claude-sonnet-4-20250514` | Pydantic AI model identifier |
| `HAROLD_LLM_TEMPERATURE` | `0.85` | Sampling temperature (0.0-2.0) |
| `HAROLD_MEMORY_BACKEND` | `in_memory` | Long-term memory backend (`in_memory` or `pgvector`) |
| `HAROLD_TRAJECTORY_BACKEND` | `in_memory` | Trajectory memory backend (`in_memory` or `neo4j`) |
| `HAROLD_NEO4J_URI` | `neo4j://localhost:7687` | Neo4j connection URI (required for neo4j) |
| `HAROLD_NEO4J_USER` | `neo4j` | Neo4j authentication username |
| `HAROLD_NEO4J_PASSWORD` | `None` | Neo4j authentication password (required for neo4j) |
| `HAROLD_PG_DSN` | `None` | PostgreSQL connection string (required for pgvector) |
| `HAROLD_EMBEDDING_MODEL` | `openai:text-embedding-3-small` | Pydantic AI embedding model identifier |
| `HAROLD_EMBEDDING_DIMENSIONS` | `1536` | Dimensionality of embedding vectors |
| `HAROLD_PHOENIX_ENABLED` | `false` | Enable Arize Phoenix tracing |
| `HAROLD_PHOENIX_ENDPOINT` | `http://127.0.0.1:6006/v1/traces` | OTLP endpoint for Phoenix |

## API Server

Harold also runs as a FastAPI server with a streaming WebSocket endpoint.

```bash
# Start the API server
uv run harold-api

# Or with auto-reload for development
uv run uvicorn harold.interfaces.api:app --reload
```

- **`GET /health`** — Health check returning status and version
- **`WebSocket /ws/{session_id}`** — Streaming improv session
- **`POST /coach/{session_id}`** — On-demand coaching feedback

The WebSocket accepts JSON messages `{"type": "message", "content": "..."}` and streams back partial text chunks (`{"type": "stream", "content": "..."}`) followed by the complete structured response (`{"type": "response", "dialogue": "...", ...}`).

## Observability

Harold integrates with [Arize Phoenix](https://docs.arize.com/phoenix) for observability. When enabled, every agent run, LLM call, tool invocation, and token usage is traced via OpenTelemetry.

```bash
# Install the Phoenix UI (one-time)
uv sync --extra phoenix

# Start Phoenix in one terminal
phoenix serve

# Enable tracing in .env
# HAROLD_PHOENIX_ENABLED=true

# Run Harold (picks up .env automatically)
uv run harold
```

Open [http://localhost:6006](http://localhost:6006) to see traces. Each agent turn produces a trace showing the LLM request/response, any tool calls, token counts, and latency breakdown.

## pgvector Memory Backend

For semantic search over past scenes and improv knowledge, Harold supports PostgreSQL with pgvector.

```bash
# Start PostgreSQL with pgvector
docker compose up -d

# Enable pgvector in .env by uncommenting:
# OPENAI_API_KEY=sk-...
# HAROLD_MEMORY_BACKEND=pgvector
# HAROLD_PG_DSN=postgresql://postgres:postgres@localhost/harold

# Seed the knowledge base with UCB improv principles (one-time)
uv run python scripts/seed_knowledge.py

# Run Harold with semantic memory
uv run harold
```

When using pgvector, the agent embeds scene summaries and knowledge entries via OpenAI's `text-embedding-3-small` model and stores them alongside their vectors. Searches use cosine similarity instead of keyword matching.

## Coaching and Pattern Analysis

Harold includes two on-demand analysis features:

- **`/coach`** — A coaching agent that examines your technique usage, recent scenes, and growth areas to provide UCB-grounded feedback. Via the API: `POST /coach/{session_id}`.
- **`/analyze`** — A pattern analyzer that discovers reusable workflow templates from your scene history. Discovered workflows are automatically injected into the scene partner's context on future scenes. Via the API: `POST /analyze`.

To enable rich trajectory tracking, use the Neo4j backend:

```bash
# Docker Compose already includes Neo4j
docker compose up -d

# Enable in .env by uncommenting:
# HAROLD_TRAJECTORY_BACKEND=neo4j
# HAROLD_NEO4J_PASSWORD=password

uv run harold
```

The Neo4j web UI is available at [http://localhost:7474](http://localhost:7474).

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

## Evaluation

Harold includes a `pydantic-evals` based evaluation suite that measures improv quality against UCB principles.

```bash
# Programmatic smoke test (requires ANTHROPIC_API_KEY)
uv run python scripts/smoke_test.py

# Full eval suite with LLM judges (requires ANTHROPIC_API_KEY)
uv run python evals/run_evals.py
```

The eval suite tests 10 cases across five UCB concepts (yes-and, strong choices, heightening, emotional truth, callbacks) using both deterministic evaluators (negation detection, information novelty) and LLM judges. When Phoenix is enabled, eval traces are visible alongside regular agent traces.
