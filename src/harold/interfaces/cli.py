"""Rich-based CLI interface for interactive improv sessions.

Provides a terminal REPL where users perform improv scenes with Harold.
Designed to be decoupled from the core agent logic so the same
dependencies and agent can be exposed via other interfaces (e.g. API).
"""

from __future__ import annotations

import asyncio
from enum import StrEnum

from pydantic_ai.messages import ModelMessage
from rich.console import Console
from rich.prompt import Prompt

from harold.agents.scene_partner import scene_partner
from harold.bootstrap import build_dependencies
from harold.config import HaroldSettings
from harold.dependencies import HaroldDependencies
from harold.observability import setup_observability

GREETING = "[bold green]Harold[/] — Your AI Improv Partner"
INSTRUCTIONS = (
    "Start a scene by saying something in character, "
    "or type 'quit' to exit.\n"
)
FAREWELL = "[dim]Scene over. Thanks for playing![/]"
USER_PROMPT_STYLE = "[bold blue]You[/]"
HAROLD_PROMPT_STYLE = "[bold yellow]Harold[/]"
STAGE_DIRECTION_STYLE = "[dim italic]"


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


def _is_quit_command(text: str) -> bool:
    """Check whether the user input is a recognized quit command.

    Strips whitespace and normalizes to lowercase before comparison.

    Args:
        text: Raw user input from the prompt.

    Returns:
        ``True`` if the input matches any ``QuitCommand`` value.
    """
    return text.strip().lower() in {cmd.value for cmd in QuitCommand}


def render_response(
    console: Console,
    dialogue: str,
    stage_direction: str | None,
) -> None:
    """Display Harold's response with appropriate Rich formatting.

    Prints the dialogue line in Harold's style, and optionally prints
    a stage direction line in italics underneath.

    Args:
        console: The Rich console instance for output.
        dialogue: Harold's spoken dialogue text.
        stage_direction: Optional physical action or stage business,
            displayed in italics if present.
    """
    console.print(f"{HAROLD_PROMPT_STYLE}: {dialogue}")
    if stage_direction:
        console.print(
            f"  {STAGE_DIRECTION_STYLE}*{stage_direction}*[/]"
        )


async def run_session(
    settings: HaroldSettings,
    dependencies: HaroldDependencies,
    console: Console,
) -> None:
    """Execute the interactive improv REPL loop.

    Prompts the user for input, sends it to the scene partner agent,
    and renders the structured response. Continues until the user
    enters a quit command.

    Args:
        settings: Application configuration, used to select the LLM model.
        dependencies: The wired dependency container for the agent.
        console: The Rich console instance for input and output.
    """
    message_history: list[ModelMessage] = []

    console.print(GREETING)
    console.print(INSTRUCTIONS)

    while True:
        user_input = Prompt.ask(USER_PROMPT_STYLE)

        if _is_quit_command(user_input):
            console.print(FAREWELL)
            break

        result = await scene_partner.run(
            user_input,
            deps=dependencies,
            message_history=message_history,
            model=settings.llm_model,
        )

        message_history.extend(result.new_messages())
        render_response(
            console,
            result.output.dialogue,
            result.output.stage_direction,
        )


async def run_cli() -> None:
    """Bootstrap settings, dependencies, and console, then start the session.

    This is the top-level async entry point that wires together all
    components before handing off to ``run_session``.
    """
    settings = HaroldSettings()
    setup_observability(settings)
    dependencies = await build_dependencies(settings)
    console = Console()
    await run_session(settings, dependencies, console)


def main() -> None:
    """Synchronous entry point for the ``harold`` console script.

    Wraps ``run_cli`` in ``asyncio.run`` for use as a setuptools
    console script entry point.
    """
    asyncio.run(run_cli())
