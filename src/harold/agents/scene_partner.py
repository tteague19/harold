"""Scene partner agent definition — the core of Harold.

Defines the module-level ``scene_partner`` agent instance configured with
UCB-grounded system prompts, structured output, and dependency injection.
Runtime variation is achieved via ``deps``, ``model`` override, and
``message_history`` parameters on ``.run()``.
"""

from __future__ import annotations

from pydantic_ai import Agent, ModelSettings, RunContext

from harold.agents.prompts import SCENE_PARTNER_SYSTEM_PROMPT
from harold.config import DEFAULT_LLM_TEMPERATURE
from harold.dependencies import HaroldDependencies
from harold.models.scene import SceneResponse

scene_partner: Agent[HaroldDependencies, SceneResponse] = Agent(
    deps_type=HaroldDependencies,
    output_type=SceneResponse,
    system_prompt=SCENE_PARTNER_SYSTEM_PROMPT,
    model_settings=ModelSettings(temperature=DEFAULT_LLM_TEMPERATURE),
)


@scene_partner.system_prompt
async def scene_context(
    ctx: RunContext[HaroldDependencies],
) -> str:
    """Inject current scene context into the system prompt.

    Appends setting and suggestion details when a scene is active,
    or a prompt to help the user start one otherwise.

    Args:
        ctx: The run context providing access to the dependency container
            and the current scene state.

    Returns:
        A string fragment appended to the system prompt describing the
        active scene, or a prompt indicating no scene is active.
    """
    if ctx.deps.current_scene:
        return (
            f"Current scene — Setting: {ctx.deps.current_scene.setting}, "
            f"Suggestion: {ctx.deps.current_scene.suggestion}"
        )
    return "No scene is currently active. Help the user start one."
