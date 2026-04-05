"""Tests for scene management functions.

Verifies scene auto-start, turn tracking, and scene ending
behavior using the in-memory dependency container.
"""

from __future__ import annotations

from harold.dependencies import HaroldDependencies
from harold.interfaces.scene_management import (
    DEFAULT_SCENE_SETTING,
    DEFAULT_SCENE_SUGGESTION,
    ensure_scene_active,
    track_turn,
)
from harold.models.scene import SceneState, Speaker


def test_ensure_scene_active_creates_scene(
    dependencies: HaroldDependencies,
) -> None:
    """Verify that ensure_scene_active creates a scene when none exists.

    Args:
        dependencies: The wired dependency container fixture.
    """
    assert dependencies.current_scene is None

    ensure_scene_active(dependencies=dependencies)

    assert dependencies.current_scene is not None
    assert dependencies.current_scene.setting == DEFAULT_SCENE_SETTING
    assert (
        dependencies.current_scene.suggestion
        == DEFAULT_SCENE_SUGGESTION
    )


def test_ensure_scene_active_preserves_existing_scene(
    dependencies: HaroldDependencies,
) -> None:
    """Verify that ensure_scene_active is a no-op when a scene exists.

    Args:
        dependencies: The wired dependency container fixture.
    """
    existing = SceneState(
        setting="a pirate ship", suggestion="treasure"
    )
    dependencies.current_scene = existing

    ensure_scene_active(dependencies=dependencies)

    assert dependencies.current_scene is existing
    assert dependencies.current_scene.setting == "a pirate ship"


def test_track_turn_appends_to_active_scene(
    dependencies: HaroldDependencies,
) -> None:
    """Verify that track_turn appends to the active scene's turns.

    Args:
        dependencies: The wired dependency container fixture.
    """
    dependencies.current_scene = SceneState(
        setting="test", suggestion="test"
    )

    track_turn(
        dependencies=dependencies,
        speaker=Speaker.USER,
        content="Hello!",
    )

    assert len(dependencies.current_scene.turns) == 1
    assert dependencies.current_scene.turns[0].speaker == Speaker.USER
    assert dependencies.current_scene.turns[0].content == "Hello!"


def test_track_turn_noop_without_scene(
    dependencies: HaroldDependencies,
) -> None:
    """Verify that track_turn is a no-op when no scene is active.

    Args:
        dependencies: The wired dependency container fixture.
    """
    assert dependencies.current_scene is None

    track_turn(
        dependencies=dependencies,
        speaker=Speaker.USER,
        content="Hello!",
    )

    assert dependencies.current_scene is None
