"""Agent tools for managing scene lifecycle.

These tools allow the agent to start and end improv scenes,
tracking state transitions through the dependency container.
"""

from __future__ import annotations

from pydantic_ai import RunContext

from harold.agents.scene_partner import scene_partner
from harold.dependencies import HaroldDependencies
from harold.models.scene import SceneState


@scene_partner.tool
async def start_new_scene(
    ctx: RunContext[HaroldDependencies],
    setting: str,
    suggestion: str,
) -> str:
    """Initialize a new improv scene with a setting and audience suggestion.

    Creates a fresh ``SceneState`` and stores it on the dependency
    container so subsequent agent turns have access to scene context.

    Args:
        ctx: The run context providing access to the dependency container.
        setting: The location or context for the new scene.
        suggestion: The audience suggestion that inspired the scene.

    Returns:
        A confirmation message including the setting and suggestion.
    """
    ctx.deps.current_scene = SceneState(
        setting=setting, suggestion=suggestion
    )
    return (
        f"Scene started in '{setting}', "
        f"inspired by the suggestion '{suggestion}'."
    )


@scene_partner.tool
async def end_scene(
    ctx: RunContext[HaroldDependencies],
    reason: str,
) -> str:
    """End the current scene when it has reached a natural conclusion.

    Clears the active scene from the dependency container. If no scene
    is active, returns a message indicating there is nothing to end.

    Args:
        ctx: The run context providing access to the dependency container.
        reason: A brief explanation of why the scene is ending.

    Returns:
        A summary message with the turn count and reason, or a message
        indicating no scene was active.
    """
    if ctx.deps.current_scene is None:
        return "No scene is currently active."

    scene = ctx.deps.current_scene
    ctx.deps.current_scene = None
    return (
        f"Scene ended after {len(scene.turns)} turns. Reason: {reason}"
    )
