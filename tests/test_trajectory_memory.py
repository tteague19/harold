"""Tests for the expanded in-memory trajectory memory backend.

Verifies scene counting, recent scene retrieval, and underused
technique detection alongside the existing frequency tracking.
"""

from __future__ import annotations

import pytest

from harold.memory.backends.in_memory import (
    CORE_TECHNIQUES,
    InMemoryTrajectoryMemory,
)
from harold.models.scene import SceneSummary

SCENE_A_ID = "scene-a"
SCENE_B_ID = "scene-b"
SCENE_C_ID = "scene-c"
DEFAULT_SETTING = "a coffee shop"
DEFAULT_SUGGESTION = "coffee"
DEFAULT_DURATION = 3


def _make_scene(
    scene_id: str,
    summary: str = "A test scene",
    techniques: list[str] | None = None,
) -> SceneSummary:
    """Create a SceneSummary with sensible defaults for testing.

    Args:
        scene_id: Unique identifier for the scene.
        summary: The narrative summary text.
        techniques: Improv techniques used, defaults to empty list.

    Returns:
        A SceneSummary populated with the given values.
    """
    return SceneSummary(
        scene_id=scene_id,
        setting=DEFAULT_SETTING,
        suggestion=DEFAULT_SUGGESTION,
        summary=summary,
        key_moments=["test moment"],
        techniques_used=techniques or [],
        duration_turns=DEFAULT_DURATION,
    )


@pytest.fixture
def memory() -> InMemoryTrajectoryMemory:
    """Provide a fresh empty trajectory memory instance.

    Returns:
        An empty InMemoryTrajectoryMemory.
    """
    return InMemoryTrajectoryMemory()


async def test_get_scene_count_empty(
    memory: InMemoryTrajectoryMemory,
) -> None:
    """Verify that an empty memory reports zero scenes.

    Args:
        memory: The in-memory trajectory memory fixture.
    """
    count = await memory.get_scene_count()
    assert count == 0


async def test_get_scene_count_after_recording(
    memory: InMemoryTrajectoryMemory,
) -> None:
    """Verify that scene count increments with each recorded scene.

    Args:
        memory: The in-memory trajectory memory fixture.
    """
    await memory.record_scene(_make_scene(SCENE_A_ID))
    await memory.record_scene(_make_scene(SCENE_B_ID))

    count = await memory.get_scene_count()
    assert count == 2


async def test_get_recent_scenes_order(
    memory: InMemoryTrajectoryMemory,
) -> None:
    """Verify that recent scenes are returned most-recent-first.

    Args:
        memory: The in-memory trajectory memory fixture.
    """
    await memory.record_scene(_make_scene(SCENE_A_ID, "First"))
    await memory.record_scene(_make_scene(SCENE_B_ID, "Second"))
    await memory.record_scene(_make_scene(SCENE_C_ID, "Third"))

    recent = await memory.get_recent_scenes(limit=2)
    assert len(recent) == 2
    assert recent[0].scene_id == SCENE_C_ID
    assert recent[1].scene_id == SCENE_B_ID


async def test_get_recent_scenes_empty(
    memory: InMemoryTrajectoryMemory,
) -> None:
    """Verify that recent scenes returns empty list when no scenes exist.

    Args:
        memory: The in-memory trajectory memory fixture.
    """
    recent = await memory.get_recent_scenes()
    assert recent == []


async def test_get_underused_techniques_all_underused(
    memory: InMemoryTrajectoryMemory,
) -> None:
    """Verify that all core techniques are reported when none are used.

    Args:
        memory: The in-memory trajectory memory fixture.
    """
    underused = await memory.get_underused_techniques(threshold=1)
    assert set(underused) == set(CORE_TECHNIQUES)


async def test_get_underused_techniques_some_used(
    memory: InMemoryTrajectoryMemory,
) -> None:
    """Verify that only genuinely underused techniques are reported.

    Args:
        memory: The in-memory trajectory memory fixture.
    """
    await memory.record_scene(
        _make_scene(SCENE_A_ID, techniques=["yes-and", "heightening"])
    )
    await memory.record_scene(
        _make_scene(SCENE_B_ID, techniques=["yes-and", "callback"])
    )

    underused = await memory.get_underused_techniques(threshold=2)
    assert "yes-and" not in underused
    assert "heightening" in underused
    assert "callback" in underused
