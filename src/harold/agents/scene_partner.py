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
