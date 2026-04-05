"""Neo4j implementation of the trajectory memory protocol.

Uses the official ``neo4j`` async driver to store scenes, turns, and
techniques as a graph. Enables rich trajectory queries for the coaching
agent: technique frequency, scene history, and underused technique
detection.
"""

from __future__ import annotations

import logging
from typing import Any

from neo4j import AsyncDriver, AsyncGraphDatabase

from harold.config import HaroldSettings
from harold.models.scene import SceneSummary
from harold.models.techniques import CORE_TECHNIQUES
from harold.models.types import (
    DEFAULT_RECENT_SCENES_LIMIT,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_UNDERUSED_THRESHOLD,
    SceneLimit,
    SearchLimit,
    TechniqueThreshold,
)
from harold.models.workflow import ImprovWorkflow

logger = logging.getLogger(__name__)

SCHEMA_CYPHER = [
    "CREATE CONSTRAINT scene_id IF NOT EXISTS "
    "FOR (s:Scene) REQUIRE s.id IS UNIQUE",
    "CREATE CONSTRAINT technique_name IF NOT EXISTS "
    "FOR (t:Technique) REQUIRE t.name IS UNIQUE",
    "CREATE INDEX scene_created_at IF NOT EXISTS "
    "FOR (s:Scene) ON (s.created_at)",
]

RECORD_SCENE_CYPHER = """\
MERGE (s:Scene {id: $scene_id})
SET s.setting = $setting,
    s.suggestion = $suggestion,
    s.summary = $summary,
    s.key_moments = $key_moments,
    s.duration_turns = $duration_turns,
    s.created_at = datetime()
WITH s
UNWIND $techniques AS tech_name
MERGE (t:Technique {name: tech_name})
MERGE (s)-[:USES_TECHNIQUE]->(t)
"""

TECHNIQUE_FREQUENCY_CYPHER = """\
MATCH (s:Scene)-[:USES_TECHNIQUE]->(t:Technique)
RETURN t.name AS name, count(s) AS count
"""

SCENE_COUNT_CYPHER = "MATCH (s:Scene) RETURN count(s) AS count"

RECENT_SCENES_CYPHER = """\
MATCH (s:Scene)
OPTIONAL MATCH (s)-[:USES_TECHNIQUE]->(t:Technique)
WITH s, collect(t.name) AS techniques
RETURN s.id AS id, s.setting AS setting,
       s.suggestion AS suggestion,
       s.summary AS summary,
       s.key_moments AS key_moments,
       s.duration_turns AS duration_turns,
       techniques
ORDER BY s.created_at DESC
LIMIT $limit
"""

STORE_WORKFLOW_CYPHER = """\
MERGE (w:Workflow {name: $name})
SET w.description = $description,
    w.scene_type = $scene_type,
    w.technique_sequence = $technique_sequence,
    w.trigger_description = $trigger_description,
    w.example_summary = $example_summary,
    w.success_count = $success_count
"""

WORKFLOWS_FOR_SCENE_CYPHER = """\
MATCH (w:Workflow)
WHERE toLower(w.trigger_description) CONTAINS toLower($search)
   OR toLower(w.scene_type) CONTAINS toLower($search)
RETURN w.name AS name, w.description AS description,
       w.scene_type AS scene_type,
       w.technique_sequence AS technique_sequence,
       w.trigger_description AS trigger_description,
       w.example_summary AS example_summary,
       w.success_count AS success_count
ORDER BY w.success_count DESC
LIMIT $limit
"""

ALL_WORKFLOWS_CYPHER = """\
MATCH (w:Workflow)
RETURN w.name AS name, w.description AS description,
       w.scene_type AS scene_type,
       w.technique_sequence AS technique_sequence,
       w.trigger_description AS trigger_description,
       w.example_summary AS example_summary,
       w.success_count AS success_count
ORDER BY w.success_count DESC
"""


def _record_to_workflow(record: Any) -> ImprovWorkflow:
    """Convert a Neo4j record to an ImprovWorkflow model.

    Args:
        record: A Neo4j record containing workflow fields.

    Returns:
        An ImprovWorkflow populated from the record.
    """
    return ImprovWorkflow(
        name=record["name"],
        description=record["description"],
        scene_type=record["scene_type"],
        technique_sequence=list(record["technique_sequence"]),
        trigger_description=record["trigger_description"],
        example_summary=record["example_summary"],
        success_count=record["success_count"],
    )


class Neo4jTrajectoryMemory:
    """Neo4j-backed trajectory memory for rich graph queries.

    Stores scenes and techniques as nodes with relationships,
    enabling pattern analysis for the coaching agent.

    Attributes:
        _driver: The async Neo4j driver instance.
    """

    def __init__(self, driver: AsyncDriver) -> None:
        """Initialize with an existing async Neo4j driver.

        Use the ``create`` factory method instead of calling this directly.

        Args:
            driver: An async Neo4j driver with verified connectivity.
        """
        self._driver = driver

    @classmethod
    async def create(
        cls, settings: HaroldSettings
    ) -> Neo4jTrajectoryMemory:
        """Async factory that creates the driver and runs schema setup.

        Args:
            settings: Application configuration containing Neo4j
                connection details.

        Returns:
            A fully initialized Neo4jTrajectoryMemory ready for use.

        Raises:
            ValueError: If ``settings.neo4j_password`` is not configured.
        """
        if settings.neo4j_password is None:
            raise ValueError(
                "HAROLD_NEO4J_PASSWORD must be set when using "
                "the neo4j trajectory backend"
            )

        driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        await driver.verify_connectivity()

        async with driver.session() as session:
            for cypher in SCHEMA_CYPHER:
                await session.run(cypher)

        logger.info(
            "Neo4j trajectory backend initialized at %s",
            settings.neo4j_uri,
        )
        return cls(driver)

    async def close(self) -> None:
        """Close the Neo4j driver and release connection pool resources.

        Should be called at application shutdown to ensure clean
        teardown of database connections.
        """
        await self._driver.close()

    async def record_scene(self, scene: SceneSummary) -> None:
        """Record a completed scene and its techniques in the graph.

        Creates a Scene node and USES_TECHNIQUE edges to Technique
        nodes (merged to avoid duplicates).

        Args:
            scene: The scene summary to record.
        """
        async with self._driver.session() as session:
            await session.run(
                RECORD_SCENE_CYPHER,
                scene_id=scene.scene_id,
                setting=scene.setting,
                suggestion=scene.suggestion,
                summary=scene.summary,
                key_moments=scene.key_moments,
                duration_turns=scene.duration_turns,
                techniques=scene.techniques_used,
            )

    async def get_technique_frequency(self) -> dict[str, int]:
        """Return frequency counts of techniques across all scenes.

        Returns:
            A mapping of technique names to the number of scenes
            in which each was used.
        """
        async with self._driver.session() as session:
            result = await session.run(TECHNIQUE_FREQUENCY_CYPHER)
            records = [record async for record in result]

        return {
            record["name"]: record["count"] for record in records
        }

    async def get_scene_count(self) -> int:
        """Return the total number of recorded scenes.

        Returns:
            The count of Scene nodes in the graph.
        """
        async with self._driver.session() as session:
            result = await session.run(SCENE_COUNT_CYPHER)
            record = await result.single()

        return record["count"] if record else 0

    async def get_recent_scenes(
        self, limit: SceneLimit = DEFAULT_RECENT_SCENES_LIMIT
    ) -> list[SceneSummary]:
        """Return the most recently recorded scenes with their techniques.

        Args:
            limit: Maximum number of scenes to return.

        Returns:
            A list of scene summaries ordered by most recent first,
            with techniques populated from graph relationships.
        """
        async with self._driver.session() as session:
            result = await session.run(
                RECENT_SCENES_CYPHER, limit=limit
            )
            records = [record async for record in result]

        return [
            SceneSummary(
                scene_id=record["id"],
                setting=record["setting"],
                suggestion=record["suggestion"],
                summary=record["summary"],
                key_moments=list(record["key_moments"]),
                techniques_used=record["techniques"] or [],
                duration_turns=record["duration_turns"],
            )
            for record in records
        ]

    async def get_underused_techniques(
        self,
        threshold: TechniqueThreshold = DEFAULT_UNDERUSED_THRESHOLD,
    ) -> list[str]:
        """Return core techniques used fewer times than the threshold.

        Compares observed technique usage against the shared
        ``CORE_TECHNIQUES`` reference list.

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
        """Persist a discovered workflow template as a graph node.

        Args:
            workflow: The workflow to store.
        """
        async with self._driver.session() as session:
            await session.run(
                STORE_WORKFLOW_CYPHER,
                name=workflow.name,
                description=workflow.description,
                scene_type=workflow.scene_type,
                technique_sequence=workflow.technique_sequence,
                trigger_description=workflow.trigger_description,
                example_summary=workflow.example_summary,
                success_count=workflow.success_count,
            )

    async def get_workflows_for_scene(
        self,
        scene_description: str,
        limit: SearchLimit = DEFAULT_SEARCH_LIMIT,
    ) -> list[ImprovWorkflow]:
        """Retrieve workflows relevant to a scene description.

        Searches workflow nodes by keyword overlap with the scene
        description in trigger descriptions and scene types.

        Args:
            scene_description: A description of the current scene
                context to match workflows against.
            limit: Maximum number of workflows to return.

        Returns:
            A list of matching workflows ordered by success count.
        """
        async with self._driver.session() as session:
            result = await session.run(
                WORKFLOWS_FOR_SCENE_CYPHER,
                search=scene_description,
                limit=limit,
            )
            records = [record async for record in result]

        return [_record_to_workflow(record) for record in records]

    async def get_all_workflows(self) -> list[ImprovWorkflow]:
        """Retrieve all stored workflow templates.

        Returns:
            A list of all stored workflows ordered by success count.
        """
        async with self._driver.session() as session:
            result = await session.run(ALL_WORKFLOWS_CYPHER)
            records = [record async for record in result]

        return [_record_to_workflow(record) for record in records]
