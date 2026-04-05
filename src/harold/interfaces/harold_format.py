"""Harold long-form improv format orchestration.

Manages the multi-scene Harold format: an opening exploration
followed by connected scenes with callbacks and thematic links.
Each function handles one discrete step to keep cognitive
complexity low.
"""

from __future__ import annotations

from pydantic_ai.messages import ModelMessage
from rich.console import Console
from rich.prompt import Prompt

from harold.config import HaroldSettings
from harold.dependencies import HaroldDependencies
from harold.interfaces.rendering import StreamingTurnResult
from harold.interfaces.scene_management import (
    end_current_scene,
    run_streaming_turn,
    track_turn,
)
from harold.models.harold_format import HaroldShow
from harold.models.scene import SceneState, Speaker
from harold.models.types import SceneNumber, SummaryTruncation

USER_PROMPT_STYLE = "[bold blue]You[/]"
END_SCENE_COMMAND = "/endscene"
SUMMARY_TRUNCATION_LENGTH: SummaryTruncation = 200


def _build_scene_prompt(
    *,
    scene_number: SceneNumber,
    setting: str,
    suggestion: str,
    has_previous_scenes: bool,
) -> str:
    """Build the opening prompt for a Harold scene.

    Args:
        scene_number: The 1-based index of this scene.
        setting: The scene's location or context.
        suggestion: The audience suggestion.
        has_previous_scenes: Whether earlier scenes exist
            for callback references.

    Returns:
        A prompt string instructing the agent to start the scene.
    """
    prompt = (
        f"Start scene {scene_number} set in '{setting}'. "
        f"The audience suggestion was '{suggestion}'."
    )
    if has_previous_scenes:
        prompt += " Make callbacks to earlier scenes."
    return prompt


def _start_scene(
    *,
    dependencies: HaroldDependencies,
    setting: str,
    suggestion: str,
) -> None:
    """Initialize a new scene on the dependency container.

    Args:
        dependencies: The wired dependency container.
        setting: The scene's location or context.
        suggestion: The audience suggestion.
    """
    dependencies.current_scene = SceneState(
        setting=setting, suggestion=suggestion
    )


async def _play_opening_beat(
    *,
    prompt: str,
    settings: HaroldSettings,
    dependencies: HaroldDependencies,
    message_history: list[ModelMessage],
    console: Console,
) -> StreamingTurnResult:
    """Play the first beat of a scene and track turns.

    Args:
        prompt: The prompt to send to the agent.
        settings: Application configuration for model selection.
        dependencies: The wired dependency container.
        message_history: The accumulated conversation history.
        console: The Rich console instance for output.

    Returns:
        The streaming turn result from the agent.
    """
    result = await run_streaming_turn(
        user_input=prompt,
        settings=settings,
        dependencies=dependencies,
        message_history=message_history,
        console=console,
    )
    message_history.extend(result.new_messages)
    track_turn(
        dependencies=dependencies,
        speaker=Speaker.USER,
        content=prompt,
    )
    track_turn(
        dependencies=dependencies,
        speaker=Speaker.HAROLD,
        content=result.output.dialogue,
    )
    return result


async def _play_scene_loop(
    *,
    settings: HaroldSettings,
    dependencies: HaroldDependencies,
    message_history: list[ModelMessage],
    console: Console,
) -> str:
    """Run the interactive scene loop until the user ends the scene.

    Args:
        settings: Application configuration for model selection.
        dependencies: The wired dependency container.
        message_history: The accumulated conversation history.
        console: The Rich console instance for input and output.

    Returns:
        The dialogue from the last turn, or empty string if the
        user ended immediately.
    """
    last_dialogue = ""
    while True:
        user_input = Prompt.ask(USER_PROMPT_STYLE)
        if user_input.strip().lower() == END_SCENE_COMMAND:
            break
        track_turn(
            dependencies=dependencies,
            speaker=Speaker.USER,
            content=user_input,
        )
        result = await run_streaming_turn(
            user_input=user_input,
            settings=settings,
            dependencies=dependencies,
            message_history=message_history,
            console=console,
        )
        message_history.extend(result.new_messages)
        track_turn(
            dependencies=dependencies,
            speaker=Speaker.HAROLD,
            content=result.output.dialogue,
        )
        last_dialogue = result.output.dialogue
    return last_dialogue


async def _run_opening(
    *,
    suggestion: str,
    settings: HaroldSettings,
    dependencies: HaroldDependencies,
    message_history: list[ModelMessage],
    console: Console,
) -> str:
    """Run the Harold opening exploration and return its summary.

    Args:
        suggestion: The audience suggestion inspiring the show.
        settings: Application configuration for model selection.
        dependencies: The wired dependency container.
        message_history: The accumulated conversation history.
        console: The Rich console instance for output.

    Returns:
        A truncated summary of the opening dialogue.
    """
    console.print("[bold]Opening exploration...[/]")
    _start_scene(
        dependencies=dependencies,
        setting="opening exploration",
        suggestion=suggestion,
    )

    result = await run_streaming_turn(
        user_input=(
            f"Let's explore the idea of '{suggestion}'. "
            f"Do a free-association opening."
        ),
        settings=settings,
        dependencies=dependencies,
        message_history=message_history,
        console=console,
    )
    message_history.extend(result.new_messages)
    await end_current_scene(
        dependencies=dependencies, console=console
    )

    return result.output.dialogue[:SUMMARY_TRUNCATION_LENGTH]


async def _run_harold_scene(
    *,
    scene_number: SceneNumber,
    suggestion: str,
    show: HaroldShow,
    settings: HaroldSettings,
    dependencies: HaroldDependencies,
    message_history: list[ModelMessage],
    console: Console,
) -> str:
    """Run a single scene within the Harold format.

    Args:
        scene_number: The 1-based index of this scene.
        suggestion: The audience suggestion.
        show: The Harold show state for callback context.
        settings: Application configuration for model selection.
        dependencies: The wired dependency container.
        message_history: The accumulated conversation history.
        console: The Rich console instance for input and output.

    Returns:
        A truncated summary of the scene.
    """
    console.print(
        f"\n[bold magenta]Scene {scene_number} "
        f"of {show.total_scenes}[/]\n"
    )
    setting = Prompt.ask("[bold]Scene setting[/]")
    _start_scene(
        dependencies=dependencies,
        setting=setting,
        suggestion=suggestion,
    )

    prompt = _build_scene_prompt(
        scene_number=scene_number,
        setting=setting,
        suggestion=suggestion,
        has_previous_scenes=bool(show.scene_summaries),
    )

    opening_result = await _play_opening_beat(
        prompt=prompt,
        settings=settings,
        dependencies=dependencies,
        message_history=message_history,
        console=console,
    )
    last_dialogue = opening_result.output.dialogue

    loop_dialogue = await _play_scene_loop(
        settings=settings,
        dependencies=dependencies,
        message_history=message_history,
        console=console,
    )
    if loop_dialogue:
        last_dialogue = loop_dialogue

    await end_current_scene(
        dependencies=dependencies, console=console
    )
    return last_dialogue[:SUMMARY_TRUNCATION_LENGTH]


async def run_harold_format(
    *,
    settings: HaroldSettings,
    dependencies: HaroldDependencies,
    console: Console,
) -> None:
    """Run a complete Harold long-form improv format.

    Prompts for a suggestion, runs an opening, then guides the
    user through connected scenes with callbacks and thematic links.

    Args:
        settings: Application configuration for model selection.
        dependencies: The wired dependency container for the agent.
        console: The Rich console instance for input and output.
    """
    suggestion = Prompt.ask(
        "[bold magenta]Audience suggestion[/]"
    )
    show = HaroldShow(suggestion=suggestion)
    dependencies.harold_show = show
    message_history: list[ModelMessage] = []

    console.print(
        f"\n[bold magenta]Harold Format[/] — "
        f"Suggestion: '{suggestion}'\n"
    )

    show.opening_summary = await _run_opening(
        suggestion=suggestion,
        settings=settings,
        dependencies=dependencies,
        message_history=message_history,
        console=console,
    )

    for scene_number in range(1, show.total_scenes + 1):
        scene_summary = await _run_harold_scene(
            scene_number=scene_number,
            suggestion=suggestion,
            show=show,
            settings=settings,
            dependencies=dependencies,
            message_history=message_history,
            console=console,
        )
        show.scene_summaries.append(scene_summary)

    dependencies.harold_show = None
    console.print(
        "[bold magenta]Harold complete! Great show.[/]\n"
    )
