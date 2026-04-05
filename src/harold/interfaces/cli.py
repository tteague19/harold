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

from harold.agents.coach import coach
from harold.agents.pattern_analyzer import pattern_analyzer
from harold.agents.scene_partner import scene_partner
from harold.bootstrap import build_dependencies
from harold.config import HaroldSettings
from harold.dependencies import HaroldDependencies
from harold.models.coaching import CoachingFeedback
from harold.models.workflow import ImprovWorkflow
from harold.observability import setup_observability

GREETING = "[bold green]Harold[/] — Your AI Improv Partner"
INSTRUCTIONS = (
    "Start a scene by saying something in character, "
    "type '/coach' for feedback, '/analyze' to discover "
    "patterns, or 'quit' to exit.\n"
)
COACH_COMMAND = "/coach"
ANALYZE_COMMAND = "/analyze"
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


def render_coaching_feedback(
    console: Console,
    feedback: CoachingFeedback,
) -> None:
    """Display coaching feedback with Rich formatting.

    Args:
        console: The Rich console instance for output.
        feedback: The structured coaching feedback to render.
    """
    console.print("\n[bold magenta]Coach's Feedback[/]\n")

    console.print("[bold green]Strengths:[/]")
    for strength in feedback.strengths:
        console.print(f"  + {strength}")

    console.print("\n[bold yellow]Growth Areas:[/]")
    for area in feedback.growth_areas:
        console.print(f"  - {area}")

    console.print("\n[bold cyan]Tips:[/]")
    for tip in feedback.specific_tips:
        console.print(f"  * {tip}")

    console.print(
        f"\n[bold]Try next:[/] {feedback.technique_suggestion}\n"
    )


async def run_coaching(
    settings: HaroldSettings,
    dependencies: HaroldDependencies,
    console: Console,
) -> None:
    """Run the coaching agent and render its feedback.

    Args:
        settings: Application configuration for model selection.
        dependencies: The wired dependency container for the agent.
        console: The Rich console instance for output.
    """
    console.print("[dim]Analyzing your improv history...[/]")
    result = await coach.run(
        "Review my improv history and give me coaching feedback.",
        deps=dependencies,
        model=settings.llm_model,
    )
    render_coaching_feedback(console, result.output)


def render_workflows(
    console: Console,
    workflows: list[ImprovWorkflow],
) -> None:
    """Display discovered workflow patterns with Rich formatting.

    Args:
        console: The Rich console instance for output.
        workflows: The list of discovered workflows to render.
    """
    console.print(
        f"\n[bold magenta]Discovered {len(workflows)} "
        f"workflow pattern(s)[/]\n"
    )
    for workflow in workflows:
        techniques = " → ".join(workflow.technique_sequence)
        console.print(f"[bold]{workflow.name}[/] ({workflow.scene_type})")
        console.print(f"  {workflow.description}")
        console.print(f"  Techniques: {techniques}")
        console.print(f"  Trigger: {workflow.trigger_description}")
        console.print(f"  Example: {workflow.example_summary}")
        console.print(f"  Used in {workflow.success_count} scene(s)\n")


async def run_analysis(
    settings: HaroldSettings,
    dependencies: HaroldDependencies,
    console: Console,
) -> None:
    """Run the pattern analyzer and store discovered workflows.

    Args:
        settings: Application configuration for model selection.
        dependencies: The wired dependency container for the agent.
        console: The Rich console instance for output.
    """
    console.print("[dim]Analyzing trajectory data for patterns...[/]")
    result = await pattern_analyzer.run(
        "Analyze my scene history and extract reusable workflow patterns.",
        deps=dependencies,
        model=settings.llm_model,
    )

    workflows = result.output
    for workflow in workflows:
        await dependencies.trajectory_memory.store_workflow(workflow)

    render_workflows(console, workflows)


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

        if user_input.strip().lower() == COACH_COMMAND:
            await run_coaching(settings, dependencies, console)
            continue

        if user_input.strip().lower() == ANALYZE_COMMAND:
            await run_analysis(settings, dependencies, console)
            continue

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
