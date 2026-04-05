"""In-memory implementations of memory protocols for MVP development.

These backends use plain Python lists and keyword matching as a stand-in
for real vector search and graph storage. They require no external
infrastructure and are suitable for development and testing.
"""

from __future__ import annotations

from collections import Counter

from harold.models.memory import KnowledgeEntry
from harold.models.scene import SceneSummary
from harold.models.techniques import CORE_TECHNIQUES
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
        """Initialize with empty scene and workflow stores."""
        self._scenes: list[SceneSummary] = []
        self._workflows: list[ImprovWorkflow] = []

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

    async def get_scene_count(self) -> int:
        """Return the total number of recorded scenes.

        Returns:
            The count of scenes stored in memory.
        """
        return len(self._scenes)

    async def get_recent_scenes(
        self, limit: SceneLimit = DEFAULT_RECENT_SCENES_LIMIT
    ) -> list[SceneSummary]:
        """Return the most recently recorded scenes.

        Args:
            limit: Maximum number of scenes to return.

        Returns:
            A list of scene summaries ordered by most recent first.
        """
        return list(reversed(self._scenes[-limit:]))

    async def get_underused_techniques(
        self,
        threshold: TechniqueThreshold = DEFAULT_UNDERUSED_THRESHOLD,
    ) -> list[str]:
        """Return core techniques used fewer times than the threshold.

        Compares observed technique usage against ``CORE_TECHNIQUES``.

        Args:
            threshold: Techniques used fewer than this many times
                are considered underused.

        Returns:
            A list of technique names that are underused or never used.
        """
        frequency = await self.get_technique_frequency()
        return [
            technique
            for technique in CORE_TECHNIQUES
            if frequency.get(technique, 0) < threshold
        ]

    async def store_workflow(
        self, workflow: ImprovWorkflow
    ) -> None:
        """Persist a discovered workflow template.

        Args:
            workflow: The workflow to append to the store.
        """
        self._workflows.append(workflow)

    async def get_workflows_for_scene(
        self,
        scene_description: str,
        limit: SearchLimit = DEFAULT_SEARCH_LIMIT,
    ) -> list[ImprovWorkflow]:
        """Retrieve workflows relevant to a scene description.

        Scores each workflow by keyword overlap between the scene
        description and the workflow's trigger description and
        scene type.

        Args:
            scene_description: A description of the current scene
                context to match workflows against.
            limit: Maximum number of workflows to return.

        Returns:
            A list of matching workflows sorted by relevance,
            excluding workflows with zero keyword overlap.
        """
        description_lower = scene_description.lower()
        tokens = description_lower.split()
        scored = [
            (
                w,
                sum(
                    t in w.trigger_description.lower()
                    or t in w.scene_type.lower()
                    for t in tokens
                ),
            )
            for w in self._workflows
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [w for w, score in scored[:limit] if score > 0]

    async def get_all_workflows(self) -> list[ImprovWorkflow]:
        """Retrieve all stored workflow templates.

        Returns:
            A list of all stored workflows.
        """
        return list(self._workflows)
