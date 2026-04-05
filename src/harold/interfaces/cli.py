"""Rich-based CLI interface for interactive improv sessions.

Provides a terminal REPL that dispatches user commands to focused
handler modules. Scene management, Harold format, character setup,
coaching, and pattern analysis each live in their own module.
"""

from __future__ import annotations

import asyncio
from enum import StrEnum

from pydantic_ai.messages import ModelMessage
from rich.console import Console
from rich.prompt import Prompt

from harold.agents.coach import coach
from harold.agents.pattern_analyzer import pattern_analyzer
from harold.bootstrap import build_dependencies
from harold.config import HaroldSettings
from harold.dependencies import HaroldDependencies
from harold.interfaces.characters import (
    rotate_character,
    setup_characters,
)
from harold.interfaces.harold_format import run_harold_format
from harold.interfaces.rendering import (
    render_coaching_feedback,
    render_workflows,
)
from harold.interfaces.scene_management import (
    end_current_scene,
    run_streaming_turn,
    track_turn,
)
from harold.models.scene import Speaker
from harold.observability import setup_observability

GREETING = "[bold green]Harold[/] — Your AI Improv Partner"
INSTRUCTIONS = (
    "Start a scene by saying something in character, "
    "type '/coach' for feedback, '/analyze' to discover "
    "patterns, '/endscene' to end a scene, '/harold' to "
    "start a Harold format, '/characters' to set up "
    "characters, or 'quit' to exit.\n"
)
USER_PROMPT_STYLE = "[bold blue]You[/]"


class QuitCommand(StrEnum):
    """Commands that terminate the CLI session.

    Attributes:
        QUIT: The full "quit" command.
        EXIT: The alternative "exit" command.
        SHORT: The single-character "q" shortcut.
    """

    QUIT = "quit"
    EXIT = "exit"
    SHORT = "q"


class SlashCommand(StrEnum):
    """Slash commands available in the REPL.

    Attributes:
        COACH: Request coaching feedback.
        ANALYZE: Discover workflow patterns from trajectory data.
        END_SCENE: End the current scene and persist it.
        HAROLD: Start a Harold long-form format.
        CHARACTERS: Set up multi-character scenes.
    """

    COACH = "/coach"
    ANALYZE = "/analyze"
    END_SCENE = "/endscene"
    HAROLD = "/harold"
    CHARACTERS = "/characters"


def _is_quit_command(text: str) -> bool:
    """Check whether the user input is a recognized quit command.

    Args:
        text: Raw user input from the prompt.

    Returns:
        ``True`` if the input matches any ``QuitCommand`` value.
    """
    return text.strip().lower() in {cmd.value for cmd in QuitCommand}


async def _handle_coach(
    *,
    settings: HaroldSettings,
    dependencies: HaroldDependencies,
    console: Console,
) -> None:
    """Run the coaching agent and render feedback.

    Args:
        settings: Application configuration for model selection.
        dependencies: The wired dependency container.
        console: The Rich console instance for output.
    """
    console.print("[dim]Analyzing your improv history...[/]")
    result = await coach.run(
        "Review my improv history and give me coaching feedback.",
        deps=dependencies,
        model=settings.llm_model,
    )
    render_coaching_feedback(console=console, feedback=result.output)


async def _handle_analyze(
    *,
    settings: HaroldSettings,
    dependencies: HaroldDependencies,
    console: Console,
) -> None:
    """Run the pattern analyzer and store discovered workflows.

    Args:
        settings: Application configuration for model selection.
        dependencies: The wired dependency container.
        console: The Rich console instance for output.
    """
    console.print(
        "[dim]Analyzing trajectory data for patterns...[/]"
    )
    result = await pattern_analyzer.run(
        "Analyze my scene history and extract reusable "
        "workflow patterns.",
        deps=dependencies,
        model=settings.llm_model,
    )
    for workflow in result.output:
        await dependencies.trajectory_memory.store_workflow(
            workflow
        )
    render_workflows(console=console, workflows=result.output)


async def _handle_slash_command(
    *,
    command: str,
    settings: HaroldSettings,
    dependencies: HaroldDependencies,
    console: Console,
) -> bool:
    """Dispatch a slash command to its handler.

    Args:
        command: The normalized slash command string.
        settings: Application configuration for model selection.
        dependencies: The wired dependency container.
        console: The Rich console instance for output.

    Returns:
        ``True`` if the command was recognized and handled.
    """
    if command == SlashCommand.COACH:
        await _handle_coach(
            settings=settings,
            dependencies=dependencies,
            console=console,
        )
        return True

    if command == SlashCommand.ANALYZE:
        await _handle_analyze(
            settings=settings,
            dependencies=dependencies,
            console=console,
        )
        return True

    if command == SlashCommand.END_SCENE:
        await end_current_scene(
            dependencies=dependencies, console=console
        )
        return True

    if command == SlashCommand.HAROLD:
        await run_harold_format(
            settings=settings,
            dependencies=dependencies,
            console=console,
        )
        return True

    if command == SlashCommand.CHARACTERS:
        setup_characters(
            dependencies=dependencies, console=console
        )
        return True

    return False


async def run_session(
    *,
    settings: HaroldSettings,
    dependencies: HaroldDependencies,
    console: Console,
) -> None:
    """Execute the interactive improv REPL loop.

    Dispatches slash commands to their handlers and sends all
    other input to the scene partner agent with streaming output.

    Args:
        settings: Application configuration for model selection.
        dependencies: The wired dependency container for the agent.
        console: The Rich console instance for input and output.
    """
    message_history: list[ModelMessage] = []

    console.print(GREETING)
    console.print(INSTRUCTIONS)

    while True:
        user_input = Prompt.ask(USER_PROMPT_STYLE)

        if _is_quit_command(user_input):
            console.print("[dim]Scene over. Thanks for playing![/]")
            break

        normalized = user_input.strip().lower()
        handled = await _handle_slash_command(
            command=normalized,
            settings=settings,
            dependencies=dependencies,
            console=console,
        )
        if handled:
            continue

        track_turn(
            dependencies=dependencies,
            speaker=Speaker.USER,
            content=user_input,
        )

        if dependencies.characters:
            rotate_character(dependencies=dependencies)

        turn_result = await run_streaming_turn(
            user_input=user_input,
            settings=settings,
            dependencies=dependencies,
            message_history=message_history,
            console=console,
        )
        message_history.extend(turn_result.new_messages)
        track_turn(
            dependencies=dependencies,
            speaker=Speaker.HAROLD,
            content=turn_result.output.dialogue,
        )


async def run_cli() -> None:
    """Bootstrap settings, dependencies, and console, then start the session."""
    settings = HaroldSettings()
    setup_observability(settings)
    dependencies = await build_dependencies(settings)
    console = Console()
    await run_session(
        settings=settings,
        dependencies=dependencies,
        console=console,
    )


def main() -> None:
    """Synchronous entry point for the ``harold`` console script."""
    asyncio.run(run_cli())
