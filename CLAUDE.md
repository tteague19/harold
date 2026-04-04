# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Harold is an AI improviser and comedy coach built with Pydantic AI and BAML. Python 3.12 project managed with uv.

## Commands

```bash
uv sync                        # Install/sync dependencies
uv run harold                  # Run the CLI
uv run harold-api              # Run the FastAPI server (localhost:8000)
uv run python -m harold        # Alternative CLI entry point
uv run pytest                  # Run tests
uv run pytest tests/test_memory.py -v  # Run a single test file
uv run ruff check src/ tests/  # Lint
uv run mypy src/               # Type check
baml generate                  # Regenerate BAML client after editing baml_src/

# pgvector memory backend
docker compose up -d                           # Start PostgreSQL + pgvector + Neo4j
uv run python scripts/seed_knowledge.py        # Seed UCB improv knowledge
HAROLD_MEMORY_BACKEND=pgvector \
  HAROLD_PG_DSN=postgresql://postgres:postgres@localhost/harold \
  uv run harold                                # Run with pgvector

# Neo4j trajectory backend
HAROLD_TRAJECTORY_BACKEND=neo4j \
  HAROLD_NEO4J_PASSWORD=password \
  uv run harold                                # Run with Neo4j trajectories

# Observability (Arize Phoenix)
uv sync --extra phoenix                        # Install Phoenix UI
phoenix serve                                  # Start Phoenix at localhost:6006
HAROLD_PHOENIX_ENABLED=true uv run harold      # Run with tracing enabled

# All HAROLD_* vars and API keys can also be set in .env (loaded automatically)
```

## Architecture

- **Pydantic AI**: Agent orchestration, tools, dependency injection, conversation history
- **BAML**: Prompt templates and standalone structured extractions (in `baml_src/`)
- **Memory**: Protocol-based backends (in-memory MVP, pgvector and Neo4j planned)
- **Interfaces**: CLI (`interfaces/cli.py`) and FastAPI WebSocket (`interfaces/api.py`)

## Code Style

- Always include docstrings on all classes, enums, and functions
- Use `StrEnum` for enumerable fields instead of `Literal` unions
- Use module-level constants instead of hard-coded default values
- Use comprehensive `Field()` constructs with validation (min_length, ge, max_length, etc.)
- Avoid extraneous inline comments and section-break comments (e.g. `# --- Section ---`) — let code and docstrings speak
- Spell out names fully (e.g. `dependencies` not `deps`)
- No inner/nested function definitions — extract to module-level functions
- No conditional imports — all imports at module top level
- Line length: 88 (Ruff/Black default)
