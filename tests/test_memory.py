"""Tests for in-memory long-term and trajectory memory backends.

Verifies storage, keyword-based retrieval, result limiting, and
technique frequency aggregation using the MVP in-memory implementations.
Uses parametrized tests for edge cases and Hypothesis for property-based
validation.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from harold.memory.backends.in_memory import (
    InMemoryLongTermMemory,
    InMemoryTrajectoryMemory,
)
from harold.models.memory import KnowledgeCategory, KnowledgeEntry
from harold.models.scene import SceneSummary


def _make_scene_summary(
    summary: str,
    *,
    scene_id: str = "scene-1",
    setting: str = "a coffee shop",
    suggestion: str = "coffee",
    techniques: list[str] | None = None,
) -> SceneSummary:
    """Create a SceneSummary with sensible defaults for testing.

    Args:
        summary: The narrative summary text.
        scene_id: Unique identifier for the scene.
        setting: The scene's location or context.
        suggestion: The audience suggestion that inspired the scene.
        techniques: Improv techniques used, defaults to empty list.

    Returns:
        A SceneSummary instance populated with the given values.
    """
    return SceneSummary(
        scene_id=scene_id,
        setting=setting,
        suggestion=suggestion,
        summary=summary,
        key_moments=["test moment"],
        techniques_used=techniques or [],
        duration_turns=3,
    )


def _make_knowledge_entry(
    content: str,
    category: KnowledgeCategory = KnowledgeCategory.UCB_PRINCIPLE,
) -> KnowledgeEntry:
    """Create a KnowledgeEntry with sensible defaults for testing.

    Args:
        content: The knowledge text content.
        category: The knowledge classification.

    Returns:
        A KnowledgeEntry instance populated with the given values.
    """
    return KnowledgeEntry(content=content, category=category)


@pytest.fixture
def long_term_memory() -> InMemoryLongTermMemory:
    """Provide a fresh empty long-term memory instance.

    Returns:
        An empty InMemoryLongTermMemory.
    """
    return InMemoryLongTermMemory()


@pytest.fixture
def trajectory_memory() -> InMemoryTrajectoryMemory:
    """Provide a fresh empty trajectory memory instance.

    Returns:
        An empty InMemoryTrajectoryMemory.
    """
    return InMemoryTrajectoryMemory()



async def test_store_and_search_scenes(
    long_term_memory: InMemoryLongTermMemory,
) -> None:
    """Verify that a stored scene can be found by keyword search.

    Args:
        long_term_memory: The in-memory long-term memory fixture.
    """
    scene = _make_scene_summary("A hilarious bakery scene on the moon")
    await long_term_memory.store_scene(scene)

    results = await long_term_memory.search_scenes("bakery")
    assert len(results) == 1
    assert results[0].scene_id == scene.scene_id


async def test_search_scenes_no_match(
    long_term_memory: InMemoryLongTermMemory,
) -> None:
    """Verify that search returns empty when no scenes match the query.

    Args:
        long_term_memory: The in-memory long-term memory fixture.
    """
    scene = _make_scene_summary("A bakery scene")
    await long_term_memory.store_scene(scene)

    results = await long_term_memory.search_scenes("spaceship")
    assert results == []


@pytest.mark.parametrize("limit", [1, 2, 3])
async def test_search_scenes_respects_limit(
    long_term_memory: InMemoryLongTermMemory,
    limit: int,
) -> None:
    """Verify that the limit parameter caps the number of scene results.

    Args:
        long_term_memory: The in-memory long-term memory fixture.
        limit: The maximum number of results to request.
    """
    for i in range(5):
        scene = _make_scene_summary(
            f"Scene {i} about improv comedy",
            scene_id=f"scene-{i}",
        )
        await long_term_memory.store_scene(scene)

    results = await long_term_memory.search_scenes("improv", limit=limit)
    assert len(results) == limit


async def test_search_scenes_ranks_by_relevance(
    long_term_memory: InMemoryLongTermMemory,
) -> None:
    """Verify that scenes with more keyword matches rank higher.

    Args:
        long_term_memory: The in-memory long-term memory fixture.
    """
    low_relevance = _make_scene_summary(
        "A scene about baking bread", scene_id="low"
    )
    high_relevance = _make_scene_summary(
        "A scene about baking bread with cakes and pies",
        scene_id="high",
    )
    await long_term_memory.store_scene(low_relevance)
    await long_term_memory.store_scene(high_relevance)

    results = await long_term_memory.search_scenes("baking cakes pies")
    assert len(results) == 2
    assert results[0].scene_id == "high"



async def test_store_and_search_knowledge(
    long_term_memory: InMemoryLongTermMemory,
) -> None:
    """Verify that stored knowledge can be retrieved by keyword.

    Args:
        long_term_memory: The in-memory long-term memory fixture.
    """
    entry = _make_knowledge_entry(
        "Always agree with your scene partner and add information"
    )
    await long_term_memory.store_knowledge(entry)

    results = await long_term_memory.search_knowledge("agree")
    assert len(results) == 1
    assert results[0].content == entry.content


@pytest.mark.parametrize("limit", [1, 2, 3])
async def test_search_knowledge_respects_limit(
    long_term_memory: InMemoryLongTermMemory,
    limit: int,
) -> None:
    """Verify that the limit parameter caps the number of knowledge results.

    Args:
        long_term_memory: The in-memory long-term memory fixture.
        limit: The maximum number of results to request.
    """
    for i in range(5):
        entry = _make_knowledge_entry(f"Improv technique number {i}")
        await long_term_memory.store_knowledge(entry)

    results = await long_term_memory.search_knowledge(
        "improv", limit=limit
    )
    assert len(results) == limit



async def test_record_and_get_technique_frequency(
    trajectory_memory: InMemoryTrajectoryMemory,
) -> None:
    """Verify that technique frequencies accumulate across recorded scenes.

    Args:
        trajectory_memory: The in-memory trajectory memory fixture.
    """
    scene_a = _make_scene_summary(
        "Scene A",
        scene_id="a",
        techniques=["yes-and", "heightening"],
    )
    scene_b = _make_scene_summary(
        "Scene B",
        scene_id="b",
        techniques=["yes-and", "callback"],
    )
    await trajectory_memory.record_scene(scene_a)
    await trajectory_memory.record_scene(scene_b)

    freq = await trajectory_memory.get_technique_frequency()
    assert freq["yes-and"] == 2
    assert freq["heightening"] == 1
    assert freq["callback"] == 1


async def test_empty_technique_frequency(
    trajectory_memory: InMemoryTrajectoryMemory,
) -> None:
    """Verify that an empty trajectory memory returns an empty frequency dict.

    Args:
        trajectory_memory: The in-memory trajectory memory fixture.
    """
    freq = await trajectory_memory.get_technique_frequency()
    assert freq == {}



@given(
    query=st.text(min_size=1, max_size=50).filter(str.strip),
)
async def test_search_scenes_never_exceeds_limit(query: str) -> None:
    """Verify that search never returns more results than the limit.

    Property: for any non-empty query string, the number of results
    is always <= the requested limit.

    Args:
        query: A randomly generated non-empty search string.
    """
    memory = InMemoryLongTermMemory()
    for i in range(10):
        scene = _make_scene_summary(
            f"Scene {i} with random words {query}",
            scene_id=f"scene-{i}",
        )
        await memory.store_scene(scene)

    limit = 3
    results = await memory.search_scenes(query, limit=limit)
    assert len(results) <= limit


@given(
    techniques=st.lists(
        st.text(min_size=1, max_size=20).filter(str.strip),
        min_size=0,
        max_size=10,
    ),
)
async def test_technique_frequency_total_matches_input(
    techniques: list[str],
) -> None:
    """Verify that total technique count equals the number of input techniques.

    Property: the sum of all frequency values equals the total number
    of technique strings recorded.

    Args:
        techniques: A randomly generated list of technique names.
    """
    memory = InMemoryTrajectoryMemory()
    scene = _make_scene_summary(
        "Test scene", techniques=techniques
    )
    await memory.record_scene(scene)

    freq = await memory.get_technique_frequency()
    assert sum(freq.values()) == len(techniques)
