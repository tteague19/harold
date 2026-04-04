"""Shared dependency wiring for all Harold interfaces.

Provides async factory functions for constructing the dependency
container from application settings. Both the CLI and API interfaces
import from this module to avoid duplicating bootstrap logic.
"""

from __future__ import annotations

from harold.config import HaroldSettings, MemoryBackend, TrajectoryBackend
from harold.dependencies import HaroldDependencies
from harold.memory.backends.in_memory import (
    InMemoryLongTermMemory,
    InMemoryTrajectoryMemory,
)
from harold.memory.backends.neo4j import Neo4jTrajectoryMemory
from harold.memory.backends.pgvector import PgVectorLongTermMemory
from harold.memory.base import LongTermMemory, TrajectoryMemory


async def create_long_term_memory(
    settings: HaroldSettings,
) -> LongTermMemory:
    """Create the long-term memory backend based on configuration.

    Args:
        settings: Application configuration controlling backend selection.

    Returns:
        A ``LongTermMemory`` implementation matching the configured backend.

    Raises:
        ValueError: If ``HAROLD_PG_DSN`` is not set when pgvector is selected.
    """
    if settings.memory_backend == MemoryBackend.PGVECTOR:
        return await PgVectorLongTermMemory.create(settings)
    return InMemoryLongTermMemory()


async def create_trajectory_memory(
    settings: HaroldSettings,
) -> TrajectoryMemory:
    """Create the trajectory memory backend based on configuration.

    Args:
        settings: Application configuration controlling backend selection.

    Returns:
        A ``TrajectoryMemory`` implementation matching the configured
        backend.

    Raises:
        ValueError: If ``HAROLD_NEO4J_PASSWORD`` is not set when neo4j
            is selected.
    """
    if settings.trajectory_backend == TrajectoryBackend.NEO4J:
        return await Neo4jTrajectoryMemory.create(settings)
    return InMemoryTrajectoryMemory()


async def build_dependencies(
    settings: HaroldSettings,
) -> HaroldDependencies:
    """Construct the dependency container from application settings.

    Selects backend implementations based on configuration and wires
    them into the dependency container.

    Args:
        settings: Application configuration controlling backend selection.

    Returns:
        A fully wired ``HaroldDependencies`` instance ready for agent use.
    """
    long_term_memory = await create_long_term_memory(settings)
    trajectory_memory = await create_trajectory_memory(settings)
    return HaroldDependencies(
        settings=settings,
        long_term_memory=long_term_memory,
        trajectory_memory=trajectory_memory,
    )
