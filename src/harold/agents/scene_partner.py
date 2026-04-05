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
from harold.models.workflow import ImprovWorkflow

DEFAULT_WORKFLOW_INJECTION_LIMIT = 3

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


def _format_workflow(workflow: ImprovWorkflow) -> str:
    """Format a single workflow template for prompt injection.

    Args:
        workflow: The workflow to format.

    Returns:
        A readable string describing the workflow pattern.
    """
    techniques = " → ".join(workflow.technique_sequence)
    return (
        f"**{workflow.name}** ({workflow.scene_type}): "
        f"{workflow.description} "
        f"Techniques: {techniques}. "
        f"Trigger: {workflow.trigger_description}"
    )


@scene_partner.system_prompt
async def workflow_context(
    ctx: RunContext[HaroldDependencies],
) -> str:
    """Inject relevant workflow templates into the system prompt.

    Queries the trajectory memory for workflows matching the current
    scene context and formats them as guidance for the agent.

    Args:
        ctx: The run context providing access to the dependency
            container and trajectory memory.

    Returns:
        A formatted string of relevant workflow templates, or an
        empty string if no workflows are available or no scene
        is active.
    """
    if not ctx.deps.current_scene:
        return ""

    scene_description = (
        f"{ctx.deps.current_scene.setting} "
        f"{ctx.deps.current_scene.suggestion}"
    )
    workflows = (
        await ctx.deps.trajectory_memory.get_workflows_for_scene(
            scene_description,
            limit=DEFAULT_WORKFLOW_INJECTION_LIMIT,
        )
    )

    if not workflows:
        return ""

    formatted = [_format_workflow(w) for w in workflows]
    return (
        "Relevant improv patterns from your history:\n"
        + "\n".join(formatted)
    )


@scene_partner.system_prompt
async def character_context(
    ctx: RunContext[HaroldDependencies],
) -> str:
    """Inject active character traits into the system prompt.

    When a character is active, instructs the agent to embody that
    character's personality and speaking style.

    Args:
        ctx: The run context providing access to the dependency
            container and the active character.

    Returns:
        A string describing the character to play, or an empty
        string if no character is active.
    """
    character = ctx.deps.active_character
    if not character:
        return ""

    return (
        f"You are playing the character '{character.name}'. "
        f"Personality: {character.personality}. "
        f"Speaking style: {character.speaking_style}. "
        f"Stay in character at all times."
    )


@scene_partner.system_prompt
async def harold_format_context(
    ctx: RunContext[HaroldDependencies],
) -> str:
    """Inject Harold format context into the system prompt.

    When a Harold show is active, provides the suggestion,
    opening themes, and summaries of previous scenes to enable
    callbacks and thematic connections.

    Args:
        ctx: The run context providing access to the dependency
            container and the Harold show state.

    Returns:
        A string describing the Harold format context, or an
        empty string if no Harold show is active.
    """
    show = ctx.deps.harold_show
    if not show:
        return ""

    parts = [
        f"You are performing a Harold format show. "
        f"Audience suggestion: '{show.suggestion}'."
    ]

    if show.opening_summary:
        parts.append(
            f"Opening themes: {show.opening_summary}"
        )

    if show.scene_summaries:
        previous = "\n".join(
            f"  Scene {i + 1}: {s}"
            for i, s in enumerate(show.scene_summaries)
        )
        parts.append(
            f"Previous scenes:\n{previous}\n"
            f"Make callbacks to earlier scenes and "
            f"explore thematic connections."
        )

    scene_number = len(show.scene_summaries) + 1
    parts.append(
        f"This is scene {scene_number} of {show.total_scenes}."
    )

    return "\n".join(parts)
