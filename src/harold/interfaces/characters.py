"""Multi-character scene setup and rotation.

Provides interactive character creation and round-robin rotation
for multi-character improv scenes. Characters inject distinct
personality traits into the scene partner's system prompt.
"""

from __future__ import annotations

from rich.console import Console
from rich.prompt import Prompt

from harold.dependencies import HaroldDependencies
from harold.models.character import Character

DONE_KEYWORD = "done"


def _prompt_for_character(console: Console) -> Character | None:
    """Prompt the user to define a single character.

    Args:
        console: The Rich console instance for input and output.

    Returns:
        A ``Character`` if the user provided details, or ``None``
        if they entered the done keyword.
    """
    name = Prompt.ask("[bold]Character name[/] (or 'done')")
    if name.strip().lower() == DONE_KEYWORD:
        return None
    personality = Prompt.ask(f"  {name}'s personality")
    style = Prompt.ask(f"  {name}'s speaking style")
    console.print(f"  Added [bold]{name}[/]\n")
    return Character(
        name=name,
        personality=personality,
        speaking_style=style,
    )


def setup_characters(
    *,
    dependencies: HaroldDependencies,
    console: Console,
) -> None:
    """Interactively define characters for multi-character scenes.

    Prompts the user to create one or more characters. Sets the
    first character as the active character.

    Args:
        dependencies: The wired dependency container to store
            characters on.
        console: The Rich console instance for input and output.
    """
    console.print("\n[bold magenta]Character Setup[/]\n")
    characters: list[Character] = []

    while True:
        character = _prompt_for_character(console=console)
        if character is None:
            break
        characters.append(character)

    if not characters:
        console.print("[dim]No characters created.[/]\n")
        return

    dependencies.characters = characters
    dependencies.active_character = characters[0]
    console.print(
        f"[bold green]{len(characters)} character(s) ready. "
        f"Active: {characters[0].name}[/]\n"
    )


def rotate_character(
    *, dependencies: HaroldDependencies
) -> None:
    """Rotate to the next character in the character list.

    Cycles through the characters list, wrapping around to the
    first character after the last.

    Args:
        dependencies: The wired dependency container with the
            character list and active character.
    """
    if not dependencies.characters:
        return
    if dependencies.active_character is None:
        dependencies.active_character = dependencies.characters[0]
        return

    current_index = dependencies.characters.index(
        dependencies.active_character
    )
    next_index = (current_index + 1) % len(dependencies.characters)
    dependencies.active_character = (
        dependencies.characters[next_index]
    )
