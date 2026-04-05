"""Scene lifecycle management for the CLI.

Provides turn tracking, scene ending with persistence, and
streaming agent turns. These functions are shared by both the
main REPL and the Harold format flow.
"""

from __future__ import annotations

from pydantic_ai.messages import ModelMessage
from rich.console import Console
from rich.live import Live
from rich.text import Text

from harold.agents.scene_partner import scene_partner
from harold.config import HaroldSettings
from harold.dependencies import HaroldDependencies
from harold.interfaces.rendering import (
    HAROLD_PROMPT_STYLE,
    STAGE_DIRECTION_STYLE,
    StreamingTurnResult,
)
from harold.models.scene import Speaker, Turn
from harold.tools.scene_tools import _summarize_and_persist

LIVE_REFRESH_PER_SECOND = 15


def track_turn(
    *,
    dependencies: HaroldDependencies,
    speaker: Speaker,
    content: str,
) -> None:
    """Append a turn to the active scene's turn history.

    No-op if no scene is currently active.

    Args:
        dependencies: The wired dependency container with the scene state.
        speaker: Who delivered this beat.
        content: The spoken or narrated content.
    """
    if dependencies.current_scene is None:
        return
    dependencies.current_scene.turns.append(
        Turn(speaker=speaker, content=content)
    )


async def end_current_scene(
    *,
    dependencies: HaroldDependencies,
    console: Console,
) -> None:
    """End the current scene, summarize it, and persist to memory.

    Args:
        dependencies: The wired dependency container with the active scene.
        console: The Rich console instance for output.
    """
    if dependencies.current_scene is None:
        console.print("[dim]No scene is currently active.[/]")
        return

    scene = dependencies.current_scene
    dependencies.current_scene = None

    if not scene.turns:
        console.print("[dim]Scene ended with no turns recorded.[/]")
        return

    console.print("[dim]Summarizing scene...[/]")
    summary = await _summarize_and_persist(
        scene=scene, dependencies=dependencies
    )
    console.print(
        f"[bold green]Scene saved:[/] {summary.summary}\n"
    )


async def run_streaming_turn(
    *,
    user_input: str,
    settings: HaroldSettings,
    dependencies: HaroldDependencies,
    message_history: list[ModelMessage],
    console: Console,
) -> StreamingTurnResult:
    """Run the scene partner agent with streaming output.

    Displays partial dialogue via Rich Live as it arrives from
    the LLM, then prints the final response with stage direction.

    Args:
        user_input: The user's scene contribution.
        settings: Application configuration for model selection.
        dependencies: The wired dependency container for the agent.
        message_history: The accumulated conversation history.
        console: The Rich console instance for live output.

    Returns:
        A ``StreamingTurnResult`` containing the structured output
        and new messages from this turn.
    """
    previous_dialogue = ""
    async with scene_partner.run_stream(
        user_input,
        deps=dependencies,
        message_history=message_history,
        model=settings.llm_model,
    ) as stream:
        with Live(
            Text(""),
            console=console,
            refresh_per_second=LIVE_REFRESH_PER_SECOND,
        ) as live:
            async for partial in stream.stream_output(
                debounce_by=None
            ):
                current = partial.dialogue
                if not current or current == previous_dialogue:
                    continue
                live.update(Text(f"Harold: {current}"))
                previous_dialogue = current

    output = await stream.get_output()
    new_messages = stream.new_messages()

    console.print(f"{HAROLD_PROMPT_STYLE}: {output.dialogue}")
    if output.stage_direction:
        console.print(
            f"  {STAGE_DIRECTION_STYLE}"
            f"*{output.stage_direction}*[/]"
        )

    return StreamingTurnResult(
        output=output, new_messages=new_messages
    )
