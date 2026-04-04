"""In-memory implementations of memory protocols for MVP development.

These backends use plain Python lists and keyword matching as a stand-in
for real vector search and graph storage. They require no external
infrastructure and are suitable for development and testing.
"""

from __future__ import annotations

from collections import Counter

from harold.models.memory import KnowledgeEntry
from harold.models.scene import SceneSummary
from harold.models.types import DEFAULT_SEARCH_LIMIT, SearchLimit, SearchQuery


class InMemoryLongTermMemory:
    """List-backed long-term memory using keyword matching.

    Stores scene summaries and knowledge entries in plain lists and
    performs relevance ranking via simple keyword overlap scoring.

    Attributes:
        _scenes: Internal list of stored scene summaries.
        _knowledge: Internal list of stored knowledge entries.
    """

    def __init__(self) -> None:
        """Initialize with empty scene and knowledge stores."""
        self._scenes: list[SceneSummary] = []
        self._knowledge: list[KnowledgeEntry] = []

    async def store_scene(self, scene: SceneSummary) -> None:
        """Persist a completed scene summary.

        Args:
            scene: The scene summary to append to the store.
        """
        self._scenes.append(scene)

    async def search_scenes(
        self,
        query: SearchQuery,
        limit: SearchLimit = DEFAULT_SEARCH_LIMIT,
    ) -> list[SceneSummary]:
        """Find scenes containing query terms in their summary.

        Scores each scene by counting how many whitespace-delimited
        query tokens appear in the scene's summary text.

        Args:
            query: Non-empty search string whose tokens are matched
                against scene summaries.
            limit: Maximum number of results to return.

        Returns:
            A list of matching scene summaries sorted by relevance
            score, excluding scenes with zero matches.
        """
        query_lower = query.lower()
        scored = [
            (s, sum(t in s.summary.lower() for t in query_lower.split()))
            for s in self._scenes
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [s for s, score in scored[:limit] if score > 0]

    async def store_knowledge(self, entry: KnowledgeEntry) -> None:
        """Persist an improv knowledge entry.

        Args:
            entry: The knowledge entry to append to the store.
        """
        self._knowledge.append(entry)

    async def search_knowledge(
        self,
        query: SearchQuery,
        limit: SearchLimit = DEFAULT_SEARCH_LIMIT,
    ) -> list[KnowledgeEntry]:
        """Find knowledge entries containing query terms.

        Scores each entry by counting how many whitespace-delimited
        query tokens appear in the entry's content text.

        Args:
            query: Non-empty search string whose tokens are matched
                against knowledge content.
            limit: Maximum number of results to return.

        Returns:
            A list of matching knowledge entries sorted by relevance
            score, excluding entries with zero matches.
        """
        query_lower = query.lower()
        scored = [
            (k, sum(t in k.content.lower() for t in query_lower.split()))
            for k in self._knowledge
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [k for k, score in scored[:limit] if score > 0]


class InMemoryTrajectoryMemory:
    """Dict-backed trajectory memory for tracking scene history.

    Records completed scenes and provides aggregate statistics
    over the improv techniques observed across all sessions.

    Attributes:
        _scenes: Internal list of recorded scene summaries.
    """

    def __init__(self) -> None:
        """Initialize with an empty scene record."""
        self._scenes: list[SceneSummary] = []

    async def record_scene(self, scene: SceneSummary) -> None:
        """Record a completed scene as a node in the trajectory graph.

        Args:
            scene: The scene summary to record.
        """
        self._scenes.append(scene)

    async def get_technique_frequency(self) -> dict[str, int]:
        """Return frequency counts of improv techniques used across all scenes.

        Returns:
            A mapping of technique names to the number of times each
            was observed. Returns an empty dict if no scenes have been
            recorded.
        """
        counter: Counter[str] = Counter()
        for scene in self._scenes:
            counter.update(scene.techniques_used)
        return dict(counter)
