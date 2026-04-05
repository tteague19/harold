"""Agent tools for managing scene lifecycle.

These tools allow the agent to start and end improv scenes,
tracking state transitions through the dependency container.
When a scene ends, it is automatically summarized via LLM and
persisted to both long-term and trajectory memory.
"""

from __future__ import annotations

from pydantic_ai import Agent, RunContext

from harold.agents.scene_partner import scene_partner
from harold.dependencies import HaroldDependencies
from harold.models.scene import SceneState, SceneSummary

SUMMARIZER_SYSTEM_PROMPT = """\
Summarize this improv scene transcript. Identify the key moments, \
the game of the scene if one emerged, and which improv techniques \
were used (e.g. yes-and, heightening, callbacks, game-of-the-scene, \
emotional-truth, strong-choices, group-mind, if-this-is-true).
"""

summarizer: Agent[None, SceneSummary] = Agent(
    output_type=SceneSummary,
    system_prompt=SUMMARIZER_SYSTEM_PROMPT,
)


def _build_transcript(scene: SceneState) -> str:
    """Build a readable transcript from a scene's turn history.

    Args:
        scene: The scene state containing the ordered turn list.

    Returns:
        A newline-delimited transcript with speaker labels.
    """
    return "\n".join(
        f"{turn.speaker.value}: {turn.content}"
        for turn in scene.turns
    )


async def _summarize_and_persist(
    scene: SceneState,
    dependencies: HaroldDependencies,
) -> SceneSummary:
    """Summarize a completed scene and persist it to memory backends.

    Generates a structured summary via LLM, stores it in long-term
    memory for semantic search, and records it in trajectory memory
    for pattern analysis.

    Args:
        scene: The completed scene state with turn history.
        dependencies: The dependency container with memory backends
            and settings.

    Returns:
        The generated SceneSummary.
    """
    transcript = _build_transcript(scene)

    prompt = (
        f"Setting: {scene.setting}\n"
        f"Suggestion: {scene.suggestion}\n\n"
        f"Transcript:\n{transcript}"
    )

    result = await summarizer.run(
        prompt,
        model=dependencies.settings.llm_model,
    )

    summary = SceneSummary(
        scene_id=scene.scene_id,
        setting=scene.setting,
        suggestion=scene.suggestion,
        summary=result.output.summary,
        key_moments=result.output.key_moments,
        techniques_used=result.output.techniques_used,
        duration_turns=len(scene.turns),
    )

    await dependencies.long_term_memory.store_scene(summary)
    await dependencies.trajectory_memory.record_scene(summary)

    return summary


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
    """End the current scene, summarize it, and persist to memory.

    Builds a transcript from the scene's turns, generates a
    structured summary via LLM, and stores it in both long-term
    and trajectory memory. Clears the active scene from the
    dependency container.

    Args:
        ctx: The run context providing access to the dependency container.
        reason: A brief explanation of why the scene is ending.

    Returns:
        A summary message with the turn count, reason, and
        confirmation that the scene was persisted.
    """
    if ctx.deps.current_scene is None:
        return "No scene is currently active."

    scene = ctx.deps.current_scene
    ctx.deps.current_scene = None

    if not scene.turns:
        return (
            f"Scene ended with no turns recorded. Reason: {reason}"
        )

    summary = await _summarize_and_persist(scene, ctx.deps)

    return (
        f"Scene ended after {len(scene.turns)} turns. "
        f"Reason: {reason}. "
        f"Summary: {summary.summary}"
    )
