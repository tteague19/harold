"""Application-wide configuration loaded from environment variables.

All settings use the ``HAROLD_`` prefix and are validated by pydantic-settings
at startup. Every external dependency defaults to the simplest option so
the MVP runs with only an Anthropic API key.

Non-prefixed variables (``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``) are
loaded into the process environment via ``dotenv.load_dotenv`` so that
third-party SDKs can discover them.
"""

from __future__ import annotations

from enum import StrEnum

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()

DEFAULT_LLM_MODEL = "anthropic:claude-sonnet-4-20250514"
DEFAULT_LLM_TEMPERATURE = 0.85
DEFAULT_PHOENIX_ENDPOINT = "http://127.0.0.1:6006/v1/traces"
DEFAULT_EMBEDDING_MODEL = "openai:text-embedding-3-small"
DEFAULT_EMBEDDING_DIMENSIONS = 1536
DEFAULT_NEO4J_URI = "neo4j://localhost:7687"
DEFAULT_NEO4J_USER = "neo4j"


class MemoryBackend(StrEnum):
    """Available storage backends for long-term vector memory.

    Attributes:
        IN_MEMORY: List-backed keyword matching (no infrastructure needed).
        PGVECTOR: PostgreSQL with pgvector extension for semantic search.
    """

    IN_MEMORY = "in_memory"
    PGVECTOR = "pgvector"


class TrajectoryBackend(StrEnum):
    """Available storage backends for trajectory graph memory.

    Attributes:
        IN_MEMORY: Dict-backed in-process storage.
        NEO4J: Neo4j graph database for rich trajectory queries.
    """

    IN_MEMORY = "in_memory"
    NEO4J = "neo4j"


class HaroldSettings(BaseSettings):
    """Application-wide configuration, loaded from ``HAROLD_``-prefixed env vars.

    Attributes:
        llm_model: Pydantic AI model identifier string.
        llm_temperature: Sampling temperature for LLM responses.
        memory_backend: Which storage backend to use for long-term memory.
        trajectory_backend: Which storage backend to use for trajectory memory.
        neo4j_uri: Neo4j connection URI for trajectory backend.
        neo4j_user: Neo4j authentication username.
        neo4j_password: Neo4j authentication password.
        pg_dsn: PostgreSQL connection string for pgvector backend.
        embedding_model: Pydantic AI embedding model identifier.
        embedding_dimensions: Dimensionality of the embedding vectors.
        phoenix_enabled: Whether to enable Arize Phoenix observability.
        phoenix_endpoint: OTLP endpoint URL for Arize Phoenix.
    """

    model_config = SettingsConfigDict(
        env_prefix="HAROLD_", env_file=".env", extra="ignore"
    )

    llm_model: str = Field(
        default=DEFAULT_LLM_MODEL,
        description="Pydantic AI model identifier",
        examples=["anthropic:claude-sonnet-4-20250514", "openai:gpt-4o"],
    )
    llm_temperature: float = Field(
        default=DEFAULT_LLM_TEMPERATURE,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for LLM responses",
    )

    memory_backend: MemoryBackend = Field(
        default=MemoryBackend.IN_MEMORY,
        description="Storage backend for long-term vector memory",
    )
    trajectory_backend: TrajectoryBackend = Field(
        default=TrajectoryBackend.IN_MEMORY,
        description="Storage backend for trajectory graph memory",
    )

    neo4j_uri: str = Field(
        default=DEFAULT_NEO4J_URI,
        description="Neo4j connection URI for trajectory backend",
        examples=["neo4j://localhost:7687"],
    )
    neo4j_user: str = Field(
        default=DEFAULT_NEO4J_USER,
        description="Neo4j authentication username",
    )
    neo4j_password: str | None = Field(
        default=None,
        description="Neo4j authentication password",
    )

    pg_dsn: str | None = Field(
        default=None,
        description="PostgreSQL connection string for pgvector backend",
        examples=["postgresql://postgres:postgres@localhost/harold"],
    )
    embedding_model: str = Field(
        default=DEFAULT_EMBEDDING_MODEL,
        description="Pydantic AI embedding model identifier",
        examples=["openai:text-embedding-3-small"],
    )
    embedding_dimensions: int = Field(
        default=DEFAULT_EMBEDDING_DIMENSIONS,
        ge=1,
        le=4096,
        description="Dimensionality of the embedding vectors",
    )

    phoenix_enabled: bool = Field(
        default=False,
        description="Enable Arize Phoenix observability",
    )
    phoenix_endpoint: str = Field(
        default=DEFAULT_PHOENIX_ENDPOINT,
        description="OTLP endpoint for Arize Phoenix",
    )
