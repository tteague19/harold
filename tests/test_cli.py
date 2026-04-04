"""Tests for CLI helper functions.

Verifies quit command detection, dependency construction, and
response rendering. Uses parametrized tests for quit command
coverage and Hypothesis for property-based validation.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st
from rich.console import Console

from harold.config import HaroldSettings
from harold.interfaces.cli import (
    QuitCommand,
    _is_quit_command,
    build_dependencies,
    render_response,
)
from harold.memory.backends.in_memory import (
    InMemoryLongTermMemory,
    InMemoryTrajectoryMemory,
)


@pytest.mark.parametrize(
    "text",
    [cmd.value for cmd in QuitCommand],
    ids=[cmd.name for cmd in QuitCommand],
)
def test_is_quit_command_recognizes_valid_commands(text: str) -> None:
    """Verify that each QuitCommand value is recognized as a quit command.

    Args:
        text: A valid quit command string.
    """
    assert _is_quit_command(text) is True


@pytest.mark.parametrize(
    "text",
    ["QUIT", "  exit  ", " Q "],
)
def test_is_quit_command_case_and_whitespace_insensitive(
    text: str,
) -> None:
    """Verify that quit detection normalizes case and strips whitespace.

    Args:
        text: A quit command with varied casing or surrounding whitespace.
    """
    assert _is_quit_command(text) is True


@pytest.mark.parametrize(
    "text",
    ["hello", "quite", "exiting", ""],
)
def test_is_quit_command_rejects_non_quit_input(text: str) -> None:
    """Verify that non-quit inputs are not treated as quit commands.

    Args:
        text: A string that should not match any quit command.
    """
    assert _is_quit_command(text) is False


@given(text=st.text(min_size=4).filter(
    lambda t: t.strip().lower() not in {cmd.value for cmd in QuitCommand}
))
def test_is_quit_command_rejects_arbitrary_text(text: str) -> None:
    """Verify that arbitrary non-quit text is never treated as a quit command.

    Property: any string not in the QuitCommand values set is rejected.

    Args:
        text: A randomly generated string that is not a quit command.
    """
    assert _is_quit_command(text) is False


async def test_build_dependencies_returns_wired_container() -> None:
    """Verify that build_dependencies returns a properly wired container."""
    settings = HaroldSettings()
    dependencies = await build_dependencies(settings)

    assert dependencies.settings is settings
    assert isinstance(
        dependencies.long_term_memory, InMemoryLongTermMemory
    )
    assert isinstance(
        dependencies.trajectory_memory, InMemoryTrajectoryMemory
    )
    assert dependencies.message_history == []
    assert dependencies.current_scene is None


def test_render_response_with_stage_direction() -> None:
    """Verify that render_response outputs both dialogue and stage direction."""
    console = Console(file=None, force_terminal=True, width=120)

    with console.capture() as capture:
        render_response(console, "Hello there!", "waves enthusiastically")

    output = capture.get()
    assert "Hello there!" in output
    assert "waves enthusiastically" in output


def test_render_response_without_stage_direction() -> None:
    """Verify that render_response omits stage direction line when None."""
    console = Console(file=None, force_terminal=True, width=120)

    with console.capture() as capture:
        render_response(console, "Hello there!", None)

    output = capture.get()
    assert "Hello there!" in output
    assert "*" not in output
