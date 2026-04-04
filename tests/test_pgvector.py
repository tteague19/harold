"""Integration tests for the pgvector long-term memory backend.

These tests require a running PostgreSQL instance with pgvector and
an ``OPENAI_API_KEY`` for embedding generation. They are marked with
``@pytest.mark.integration`` and skipped by default.

Run with: ``uv run pytest tests/test_pgvector.py -v -m integration``
"""

from __future__ import annotations

import pytest

from harold.config import HaroldSettings
from harold.memory.backends.pgvector import PgVectorLongTermMemory
from harold.models.memory import KnowledgeCategory, KnowledgeEntry
from harold.models.scene import SceneSummary

INTEGRATION_SCENE_ID = "integration-test-scene"
INTEGRATION_SETTING = "a test laboratory"
INTEGRATION_SUGGESTION = "science"
INTEGRATION_SUMMARY = "Two scientists discover their experiment has become sentient"
INTEGRATION_KEY_MOMENT = "the beaker started talking"
INTEGRATION_TECHNIQUE = "yes-and"
INTEGRATION_DURATION_TURNS = 5

INTEGRATION_KNOWLEDGE_CONTENT = (
    "Always agree with your scene partner and build on their offer"
)


@pytest.fixture
def integration_settings() -> HaroldSettings:
    """Provide settings configured for pgvector integration testing.

    Returns:
        HaroldSettings with pgvector backend and DSN from environment.
    """
    return HaroldSettings(memory_backend="pgvector")


@pytest.fixture
def scene_summary() -> SceneSummary:
    """Provide a sample scene summary for integration tests.

    Returns:
        A SceneSummary with deterministic test values.
    """
    return SceneSummary(
        scene_id=INTEGRATION_SCENE_ID,
        setting=INTEGRATION_SETTING,
        suggestion=INTEGRATION_SUGGESTION,
        summary=INTEGRATION_SUMMARY,
        key_moments=[INTEGRATION_KEY_MOMENT],
        techniques_used=[INTEGRATION_TECHNIQUE],
        duration_turns=INTEGRATION_DURATION_TURNS,
    )


@pytest.fixture
def knowledge_entry() -> KnowledgeEntry:
    """Provide a sample knowledge entry for integration tests.

    Returns:
        A KnowledgeEntry with deterministic test values.
    """
    return KnowledgeEntry(
        content=INTEGRATION_KNOWLEDGE_CONTENT,
        category=KnowledgeCategory.UCB_PRINCIPLE,
        source="integration test",
    )


@pytest.mark.integration
async def test_pgvector_store_and_search_scenes(
    integration_settings: HaroldSettings,
    scene_summary: SceneSummary,
) -> None:
    """Verify that a scene can be stored and retrieved via semantic search.

    Args:
        integration_settings: Settings configured for pgvector.
        scene_summary: The sample scene summary to store and search for.
    """
    memory = await PgVectorLongTermMemory.create(integration_settings)

    await memory.store_scene(scene_summary)
    results = await memory.search_scenes("sentient experiment science")

    assert len(results) >= 1
    assert any(r.scene_id == INTEGRATION_SCENE_ID for r in results)


@pytest.mark.integration
async def test_pgvector_store_and_search_knowledge(
    integration_settings: HaroldSettings,
    knowledge_entry: KnowledgeEntry,
) -> None:
    """Verify that a knowledge entry can be stored and retrieved via semantic search.

    Args:
        integration_settings: Settings configured for pgvector.
        knowledge_entry: The sample knowledge entry to store and search for.
    """
    memory = await PgVectorLongTermMemory.create(integration_settings)

    await memory.store_knowledge(knowledge_entry)
    results = await memory.search_knowledge("agreement improv partner")

    assert len(results) >= 1
    assert any(
        r.content == INTEGRATION_KNOWLEDGE_CONTENT for r in results
    )
