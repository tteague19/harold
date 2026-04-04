"""Agent tools for the coaching agent to query trajectory memory.

These tools are registered on the coaching agent and provide access
to the user's improv history for pattern analysis and feedback
generation.
"""

from __future__ import annotations

from pydantic_ai import RunContext

from harold.agents.coach import coach
from harold.dependencies import HaroldDependencies
from harold.models.types import (
    DEFAULT_RECENT_SCENES_LIMIT,
    DEFAULT_UNDERUSED_THRESHOLD,
)


@coach.tool
async def get_technique_summary(
    ctx: RunContext[HaroldDependencies],
) -> str:
    """Retrieve a summary of technique usage across all recorded scenes.

    Returns a formatted string showing each technique and how many
    times it has been used, plus the total scene count for context.

    Args:
        ctx: The run context providing access to the trajectory
            memory backend.

    Returns:
        A formatted summary of technique frequencies and scene count.
    """
    frequency = await ctx.deps.trajectory_memory.get_technique_frequency()
    scene_count = await ctx.deps.trajectory_memory.get_scene_count()

    if not frequency:
        return f"No techniques recorded across {scene_count} scenes."

    lines = [f"Technique usage across {scene_count} scenes:"]
    for technique, count in sorted(
        frequency.items(), key=lambda pair: pair[1], reverse=True
    ):
        lines.append(f"  {technique}: {count}")
    return "\n".join(lines)


@coach.tool
async def get_recent_scene_summaries(
    ctx: RunContext[HaroldDependencies],
) -> str:
    """Retrieve summaries of the most recent scenes for review.

    Args:
        ctx: The run context providing access to the trajectory
            memory backend.

    Returns:
        A formatted string with recent scene summaries, or a message
        indicating no scenes have been recorded.
    """
    scenes = await ctx.deps.trajectory_memory.get_recent_scenes(
        limit=DEFAULT_RECENT_SCENES_LIMIT
    )

    if not scenes:
        return "No scenes have been recorded yet."

    lines = []
    for scene in scenes:
        lines.append(
            f"Scene '{scene.setting}' (suggestion: '{scene.suggestion}'): "
            f"{scene.summary}"
        )
    return "\n".join(lines)


@coach.tool
async def identify_growth_areas(
    ctx: RunContext[HaroldDependencies],
) -> str:
    """Identify improv techniques the user has underused or never tried.

    Compares the user's technique history against core UCB techniques
    to find gaps in their practice.

    Args:
        ctx: The run context providing access to the trajectory
            memory backend.

    Returns:
        A formatted string listing underused techniques, or a message
        indicating the user has good technique coverage.
    """
    underused = await ctx.deps.trajectory_memory.get_underused_techniques(
        threshold=DEFAULT_UNDERUSED_THRESHOLD
    )

    if not underused:
        return (
            "Good technique coverage! The performer has used all "
            "core techniques at least a few times."
        )

    return (
        "Techniques the performer could explore more: "
        + ", ".join(underused)
    )
