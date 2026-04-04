"""Agent tools for retrieving from long-term memory.

These tools are registered on the scene partner agent and invoked by
the LLM when it determines that recalling past scenes or improv
knowledge would benefit the current interaction.
"""

from __future__ import annotations

from pydantic_ai import RunContext

from harold.agents.scene_partner import scene_partner
from harold.dependencies import HaroldDependencies
from harold.models.types import (
    DEFAULT_KNOWLEDGE_RECALL_LIMIT,
    DEFAULT_SCENE_RECALL_LIMIT,
)


@scene_partner.tool
async def recall_similar_scenes(
    ctx: RunContext[HaroldDependencies],
    topic: str,
) -> list[str]:
    """Search past scenes for similar themes or situations to callback to.

    Args:
        ctx: The run context providing access to the dependency container
            and its long-term memory backend.
        topic: A keyword or phrase describing the theme to search for.

    Returns:
        A list of summary strings from scenes matching the topic.
    """
    results = await ctx.deps.long_term_memory.search_scenes(
        topic, limit=DEFAULT_SCENE_RECALL_LIMIT
    )
    return [scene.summary for scene in results]


@scene_partner.tool
async def recall_improv_knowledge(
    ctx: RunContext[HaroldDependencies],
    concept: str,
) -> str:
    """Look up improv principles or techniques relevant to the current scene.

    Args:
        ctx: The run context providing access to the dependency container
            and its long-term memory backend.
        concept: A keyword or phrase describing the improv concept to
            look up.

    Returns:
        A newline-joined string of matching knowledge entry contents,
        or an empty string if no matches are found.
    """
    results = await ctx.deps.long_term_memory.search_knowledge(
        concept, limit=DEFAULT_KNOWLEDGE_RECALL_LIMIT
    )
    return "\n".join(entry.content for entry in results)
