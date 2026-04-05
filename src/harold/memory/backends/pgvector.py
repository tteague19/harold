"""PostgreSQL + pgvector implementation of the long-term memory protocol.

Uses ``asyncpg`` for async database access and the ``pgvector`` extension
for cosine similarity search over embedding vectors. Designed to be
created via the async ``create`` factory method.
"""

from __future__ import annotations

import json
import logging

import asyncpg
from pgvector.asyncpg import register_vector

from harold.config import HaroldSettings
from harold.memory.embeddings import embed_text
from harold.models.memory import KnowledgeCategory, KnowledgeEntry
from harold.models.scene import SceneSummary
from harold.models.types import DEFAULT_SEARCH_LIMIT, SearchLimit, SearchQuery

logger = logging.getLogger(__name__)

SCHEMA_SQL = """\
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS scenes (
    id TEXT PRIMARY KEY,
    setting TEXT NOT NULL,
    suggestion TEXT NOT NULL,
    summary TEXT NOT NULL,
    key_moments JSONB NOT NULL,
    techniques_used JSONB NOT NULL,
    duration_turns INTEGER NOT NULL,
    embedding vector({dimensions}),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS knowledge (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    category TEXT NOT NULL,
    source TEXT,
    embedding vector({dimensions}),
    created_at TIMESTAMPTZ DEFAULT NOW()
);
"""

INSERT_SCENE_SQL = """\
INSERT INTO scenes (id, setting, suggestion, summary, key_moments,
                    techniques_used, duration_turns, embedding)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
ON CONFLICT (id) DO UPDATE SET
    summary = EXCLUDED.summary,
    embedding = EXCLUDED.embedding
"""

SEARCH_SCENES_SQL = """\
SELECT id, setting, suggestion, summary, key_moments,
       techniques_used, duration_turns
FROM scenes
ORDER BY embedding <=> $1
LIMIT $2
"""

INSERT_KNOWLEDGE_SQL = """\
INSERT INTO knowledge (content, category, source, embedding)
VALUES ($1, $2, $3, $4)
"""

SEARCH_KNOWLEDGE_SQL = """\
SELECT content, category, source
FROM knowledge
ORDER BY embedding <=> $1
LIMIT $2
"""


async def _register_vector_on_connection(
    conn: asyncpg.Connection,
) -> None:
    """Register pgvector types on a new asyncpg connection.

    Passed as the ``init`` callback to ``asyncpg.create_pool`` so that
    every connection in the pool can handle vector columns.

    Args:
        conn: The asyncpg connection to initialize.
    """
    await register_vector(conn)


class PgVectorLongTermMemory:
    """PostgreSQL + pgvector backend for long-term semantic memory.

    Stores scene summaries and knowledge entries alongside their
    embedding vectors, and retrieves them via cosine similarity search.

    Attributes:
        _pool: The asyncpg connection pool for database access.
        _settings: Application configuration for embedding generation.
    """

    def __init__(
        self, pool: asyncpg.Pool, settings: HaroldSettings
    ) -> None:
        """Initialize with an existing connection pool and settings.

        Use the ``create`` factory method instead of calling this directly.

        Args:
            pool: An asyncpg connection pool with pgvector registered.
            settings: Application configuration for embedding generation.
        """
        self._pool = pool
        self._settings = settings

    async def close(self) -> None:
        """Close the connection pool and release database resources.

        Should be called at application shutdown to ensure clean
        teardown of database connections.
        """
        await self._pool.close()

    @classmethod
    async def create(cls, settings: HaroldSettings) -> PgVectorLongTermMemory:
        """Async factory that creates the pool and runs migrations.

        Args:
            settings: Application configuration containing ``pg_dsn``
                and ``embedding_dimensions``.

        Returns:
            A fully initialized PgVectorLongTermMemory ready for use.

        Raises:
            ValueError: If ``settings.pg_dsn`` is not configured.
        """
        if settings.pg_dsn is None:
            raise ValueError(
                "HAROLD_PG_DSN must be set when using the pgvector backend"
            )

        pool = await asyncpg.create_pool(
            settings.pg_dsn, init=_register_vector_on_connection
        )

        async with pool.acquire() as conn:
            schema = SCHEMA_SQL.format(dimensions=settings.embedding_dimensions)
            await conn.execute(schema)

        logger.info("PgVector backend initialized with DSN: %s", settings.pg_dsn)
        return cls(pool, settings)

    async def store_scene(self, scene: SceneSummary) -> None:
        """Persist a completed scene summary with its embedding vector.

        Embeds the scene's summary text and stores all fields in the
        ``scenes`` table. Uses upsert to handle duplicate scene IDs.

        Args:
            scene: The scene summary to store.
        """
        embedding = await embed_text(scene.summary, self._settings)

        async with self._pool.acquire() as conn:
            await conn.execute(
                INSERT_SCENE_SQL,
                scene.scene_id,
                scene.setting,
                scene.suggestion,
                scene.summary,
                json.dumps(scene.key_moments),
                json.dumps(scene.techniques_used),
                scene.duration_turns,
                embedding,
            )

    async def search_scenes(
        self,
        query: SearchQuery,
        limit: SearchLimit = DEFAULT_SEARCH_LIMIT,
    ) -> list[SceneSummary]:
        """Find scenes semantically similar to the query via cosine similarity.

        Embeds the query text and searches the ``scenes`` table using
        pgvector's ``<=>`` cosine distance operator.

        Args:
            query: Non-empty search string to embed and match.
            limit: Maximum number of results to return.

        Returns:
            A list of matching scene summaries ordered by cosine similarity.
        """
        query_embedding = await embed_text(query, self._settings)

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                SEARCH_SCENES_SQL, query_embedding, limit
            )

        return [
            SceneSummary(
                scene_id=row["id"],
                setting=row["setting"],
                suggestion=row["suggestion"],
                summary=row["summary"],
                key_moments=json.loads(row["key_moments"]),
                techniques_used=json.loads(row["techniques_used"]),
                duration_turns=row["duration_turns"],
            )
            for row in rows
        ]

    async def store_knowledge(self, entry: KnowledgeEntry) -> None:
        """Persist an improv knowledge entry with its embedding vector.

        Embeds the entry's content text and stores all fields in the
        ``knowledge`` table.

        Args:
            entry: The knowledge entry to store.
        """
        embedding = await embed_text(entry.content, self._settings)

        async with self._pool.acquire() as conn:
            await conn.execute(
                INSERT_KNOWLEDGE_SQL,
                entry.content,
                entry.category.value,
                entry.source,
                embedding,
            )

    async def search_knowledge(
        self,
        query: SearchQuery,
        limit: SearchLimit = DEFAULT_SEARCH_LIMIT,
    ) -> list[KnowledgeEntry]:
        """Find knowledge entries semantically similar to the query.

        Embeds the query text and searches the ``knowledge`` table using
        pgvector's ``<=>`` cosine distance operator.

        Args:
            query: Non-empty search string to embed and match.
            limit: Maximum number of results to return.

        Returns:
            A list of matching knowledge entries ordered by cosine similarity.
        """
        query_embedding = await embed_text(query, self._settings)

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                SEARCH_KNOWLEDGE_SQL, query_embedding, limit
            )

        return [
            KnowledgeEntry(
                content=row["content"],
                category=KnowledgeCategory(row["category"]),
                source=row["source"],
            )
            for row in rows
        ]
