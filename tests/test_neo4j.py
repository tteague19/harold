"""Integration tests for the Neo4j trajectory memory backend.

These tests require a running Neo4j instance. They are marked with
``@pytest.mark.integration`` and skipped by default.

Run with: ``uv run pytest tests/test_neo4j.py -v -m integration``
"""

from __future__ import annotations

import pytest

from harold.config import HaroldSettings
from harold.memory.backends.neo4j import Neo4jTrajectoryMemory
from harold.models.scene import SceneSummary

INTEGRATION_SCENE_ID = "neo4j-test-scene"
INTEGRATION_SETTING = "a test laboratory"
INTEGRATION_SUGGESTION = "science"
INTEGRATION_SUMMARY = "Two scientists discover gravity works backwards"
INTEGRATION_KEY_MOMENT = "the apple fell up"
INTEGRATION_TECHNIQUES = ["yes-and", "heightening"]
INTEGRATION_DURATION = 5


@pytest.fixture
def integration_settings() -> HaroldSettings:
    """Provide settings configured for Neo4j integration testing.

    Returns:
        HaroldSettings with neo4j trajectory backend and credentials
        from environment.
    """
    return HaroldSettings(trajectory_backend="neo4j")


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
        techniques_used=INTEGRATION_TECHNIQUES,
        duration_turns=INTEGRATION_DURATION,
    )


@pytest.mark.integration
async def test_neo4j_record_and_count_scenes(
    integration_settings: HaroldSettings,
    scene_summary: SceneSummary,
) -> None:
    """Verify that a scene can be recorded and counted in Neo4j.

    Args:
        integration_settings: Settings configured for Neo4j.
        scene_summary: The sample scene summary to record.
    """
    memory = await Neo4jTrajectoryMemory.create(integration_settings)

    await memory.record_scene(scene_summary)
    count = await memory.get_scene_count()

    assert count >= 1


@pytest.mark.integration
async def test_neo4j_technique_frequency(
    integration_settings: HaroldSettings,
    scene_summary: SceneSummary,
) -> None:
    """Verify that technique frequency is tracked across scenes.

    Args:
        integration_settings: Settings configured for Neo4j.
        scene_summary: The sample scene summary to record.
    """
    memory = await Neo4jTrajectoryMemory.create(integration_settings)

    await memory.record_scene(scene_summary)
    frequency = await memory.get_technique_frequency()

    assert "yes-and" in frequency
    assert "heightening" in frequency


@pytest.mark.integration
async def test_neo4j_recent_scenes(
    integration_settings: HaroldSettings,
    scene_summary: SceneSummary,
) -> None:
    """Verify that recent scenes are retrievable from Neo4j.

    Args:
        integration_settings: Settings configured for Neo4j.
        scene_summary: The sample scene summary to record.
    """
    memory = await Neo4jTrajectoryMemory.create(integration_settings)

    await memory.record_scene(scene_summary)
    recent = await memory.get_recent_scenes(limit=5)

    assert len(recent) >= 1
    assert any(s.scene_id == INTEGRATION_SCENE_ID for s in recent)
