"""Rich rendering functions for CLI output.

Provides formatting helpers for scene responses, coaching feedback,
and workflow patterns. Shared across the CLI REPL and Harold format
modules.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai.messages import ModelMessage
from rich.console import Console

from harold.models.coaching import CoachingFeedback
from harold.models.scene import SceneResponse
from harold.models.workflow import ImprovWorkflow

HAROLD_PROMPT_STYLE = "[bold yellow]Harold[/]"
STAGE_DIRECTION_STYLE = "[dim italic]"


@dataclass
class StreamingTurnResult:
    """Result of a single streaming agent turn.

    Bundles the structured output with the new messages generated
    during the turn, so callers don't need to manage both separately.

    Attributes:
        output: The agent's structured SceneResponse.
        new_messages: Messages generated during this turn, to be
            appended to the conversation history.
    """

    output: SceneResponse
    new_messages: list[ModelMessage]


def render_response(
    console: Console,
    dialogue: str,
    stage_direction: str | None,
) -> None:
    """Display Harold's response with appropriate Rich formatting.

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
        console.print(
            f"[bold]{workflow.name}[/] ({workflow.scene_type})"
        )
        console.print(f"  {workflow.description}")
        console.print(f"  Techniques: {techniques}")
        console.print(f"  Trigger: {workflow.trigger_description}")
        console.print(f"  Example: {workflow.example_summary}")
        console.print(
            f"  Used in {workflow.success_count} scene(s)\n"
        )
