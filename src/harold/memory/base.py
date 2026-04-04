"""Abstract interfaces for Harold's memory systems.

Defines ``Protocol`` classes for long-term vector memory and trajectory
graph memory. Backends implement these protocols without inheritance,
enabling easy swapping between in-memory, pgvector, and Neo4j stores.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from harold.models.memory import KnowledgeEntry
from harold.models.scene import SceneSummary
from harold.models.types import DEFAULT_SEARCH_LIMIT, SearchLimit, SearchQuery


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
