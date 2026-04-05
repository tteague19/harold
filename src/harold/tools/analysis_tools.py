"""Agent tools for the pattern analyzer to query trajectory memory.

These tools are registered on the pattern analyzer agent via the
``@pattern_analyzer.tool`` decorator and provide access to the user's
scene history for discovering reusable workflow patterns.
"""

from __future__ import annotations

from pydantic_ai import RunContext

from harold.agents.pattern_analyzer import pattern_analyzer
from harold.dependencies import HaroldDependencies

ANALYSIS_SCENE_LIMIT = 10


def _format_scene_line(
    scene_setting: str,
    suggestion: str,
    summary: str,
    techniques: list[str],
) -> str:
    """Format a single scene into a readable summary line.

    Args:
        scene_setting: The scene's location or context.
        suggestion: The audience suggestion.
        summary: The scene narrative summary.
        techniques: Techniques used in the scene.

    Returns:
        A formatted single-line scene summary.
    """
    technique_text = ", ".join(techniques) or "none"
    return (
        f"Scene '{scene_setting}' "
        f"(suggestion: '{suggestion}'): "
        f"{summary} "
        f"[techniques: {technique_text}]"
    )


@pattern_analyzer.tool
async def get_scene_history_for_analysis(
    ctx: RunContext[HaroldDependencies],
) -> str:
    """Retrieve recent scene summaries for pattern discovery.

    Returns a formatted string of the most recent scenes including
    their settings, suggestions, summaries, and techniques used.

    Args:
        ctx: The run context providing access to the trajectory
            memory backend.

    Returns:
        A formatted summary of recent scenes, or a message
        indicating no scenes have been recorded.
    """
    scenes = await ctx.deps.trajectory_memory.get_recent_scenes(
        limit=ANALYSIS_SCENE_LIMIT
    )

    if not scenes:
        return "No scenes have been recorded yet."

    lines = [
        _format_scene_line(
            s.setting, s.suggestion, s.summary, s.techniques_used
        )
        for s in scenes
    ]
    return "\n".join(lines)


@pattern_analyzer.tool
async def get_technique_usage_for_analysis(
    ctx: RunContext[HaroldDependencies],
) -> str:
    """Retrieve technique frequency data for pattern discovery.

    Args:
        ctx: The run context providing access to the trajectory
            memory backend.

    Returns:
        A formatted summary of technique usage frequencies and
        total scene count.
    """
    frequency = (
        await ctx.deps.trajectory_memory.get_technique_frequency()
    )
    scene_count = await ctx.deps.trajectory_memory.get_scene_count()

    if not frequency:
        return f"No techniques recorded across {scene_count} scenes."

    sorted_items = sorted(
        frequency.items(), key=lambda pair: pair[1], reverse=True
    )
    detail_lines = [
        f"  {technique}: {count}"
        for technique, count in sorted_items
    ]
    return "\n".join(
        [f"Technique usage across {scene_count} scenes:"]
        + detail_lines
    )
