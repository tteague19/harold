"""Abstract interfaces for Harold's memory systems.

Defines ``Protocol`` classes for long-term vector memory and trajectory
graph memory. Backends implement these protocols without inheritance,
enabling easy swapping between in-memory, pgvector, and Neo4j stores.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from harold.models.memory import KnowledgeEntry
from harold.models.scene import SceneSummary
from harold.models.types import (
    DEFAULT_RECENT_SCENES_LIMIT,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_UNDERUSED_THRESHOLD,
    SceneLimit,
    SearchLimit,
    SearchQuery,
    TechniqueThreshold,
)
from harold.models.workflow import ImprovWorkflow


@runtime_checkable
class LongTermMemory(Protocol):
    """Interface for persistent vector-searchable memory.

    Implementations must provide methods for storing and retrieving
    both scene summaries and improv knowledge entries.
    """

    async def store_scene(self, scene: SceneSummary) -> None:
        """Persist a completed scene summary.

        Args:
            scene: The scene summary to store.
        """
        ...

    async def search_scenes(
        self,
        query: SearchQuery,
        limit: SearchLimit = DEFAULT_SEARCH_LIMIT,
    ) -> list[SceneSummary]:
        """Find scenes similar to the query.

        Args:
            query: Non-empty search string to match against stored scenes.
            limit: Maximum number of results to return.

        Returns:
            A list of matching scene summaries, ordered by relevance.
        """
        ...

    async def store_knowledge(self, entry: KnowledgeEntry) -> None:
        """Persist an improv knowledge entry.

        Args:
            entry: The knowledge entry to store.
        """
        ...

    async def search_knowledge(
        self,
        query: SearchQuery,
        limit: SearchLimit = DEFAULT_SEARCH_LIMIT,
    ) -> list[KnowledgeEntry]:
        """Find knowledge entries relevant to the query.

        Args:
            query: Non-empty search string to match against stored knowledge.
            limit: Maximum number of results to return.

        Returns:
            A list of matching knowledge entries, ordered by relevance.
        """
        ...


@runtime_checkable
class TrajectoryMemory(Protocol):
    """Interface for graph-based trajectory and interaction traces.

    Implementations must provide methods for recording completed scenes
    and querying aggregate patterns across the user's improv history.
    """

    async def record_scene(self, scene: SceneSummary) -> None:
        """Record a completed scene as a node in the trajectory graph.

        Args:
            scene: The scene summary to record.
        """
        ...

    async def get_technique_frequency(self) -> dict[str, int]:
        """Return frequency counts of improv techniques used across all scenes.

        Returns:
            A mapping of technique names to the number of times each
            was observed.
        """
        ...

    async def get_scene_count(self) -> int:
        """Return the total number of recorded scenes.

        Returns:
            The count of scenes in the trajectory store.
        """
        ...

    async def get_recent_scenes(
        self, limit: SceneLimit = DEFAULT_RECENT_SCENES_LIMIT
    ) -> list[SceneSummary]:
        """Return the most recently recorded scenes.

        Args:
            limit: Maximum number of scenes to return.

        Returns:
            A list of scene summaries ordered by most recent first.
        """
        ...

    async def get_underused_techniques(
        self,
        threshold: TechniqueThreshold = DEFAULT_UNDERUSED_THRESHOLD,
    ) -> list[str]:
        """Return techniques used fewer times than the threshold.

        Compares observed technique usage against the shared
        ``CORE_TECHNIQUES`` list from ``harold.models.techniques``
        (e.g. "yes-and", "heightening", "callback"). Implementations
        should import this list rather than defining their own.

        Args:
            threshold: Techniques used fewer than this many times
                are considered underused.

        Returns:
            A list of technique names that are underused or never
            used, in the order they appear in ``CORE_TECHNIQUES``.
        """
        ...

    async def store_workflow(
        self, workflow: ImprovWorkflow
    ) -> None:
        """Persist a discovered workflow template.

        Args:
            workflow: The workflow to store.
        """
        ...

    async def get_workflows_for_scene(
        self,
        scene_description: str,
        limit: SearchLimit = DEFAULT_SEARCH_LIMIT,
    ) -> list[ImprovWorkflow]:
        """Retrieve workflows relevant to a scene description.

        Args:
            scene_description: A description of the current scene
                context to match workflows against.
            limit: Maximum number of workflows to return.

        Returns:
            A list of matching workflows ordered by relevance.
        """
        ...

    async def get_all_workflows(self) -> list[ImprovWorkflow]:
        """Retrieve all stored workflow templates.

        Returns:
            A list of all stored workflows.
        """
        ...
