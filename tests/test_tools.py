"""Tests for scene lifecycle tools.

Verifies that start_new_scene and end_scene correctly mutate the
current_scene state on the dependency container. Tests call the
tool functions directly with a mock RunContext to validate behavior
in isolation.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from harold.dependencies import HaroldDependencies
from harold.models.scene import SceneState
from harold.tools.scene_tools import end_scene, start_new_scene


@pytest.fixture
def mock_context(
    dependencies: HaroldDependencies,
) -> MagicMock:
    """Provide a mock RunContext whose deps attribute is the real container.

    Args:
        dependencies: The wired dependency container fixture.

    Returns:
        A MagicMock with ``deps`` pointing to the real HaroldDependencies.
    """
    ctx = MagicMock()
    ctx.deps = dependencies
    return ctx


async def test_start_new_scene_sets_state(
    mock_context: MagicMock,
    dependencies: HaroldDependencies,
) -> None:
    """Verify that start_new_scene populates current_scene on dependencies.

    Args:
        mock_context: The mock RunContext fixture.
        dependencies: The wired dependency container fixture.
    """
    assert dependencies.current_scene is None

    result = await start_new_scene(
        mock_context, setting="a pirate ship", suggestion="treasure"
    )

    assert dependencies.current_scene is not None
    assert dependencies.current_scene.setting == "a pirate ship"
    assert dependencies.current_scene.suggestion == "treasure"
    assert "pirate ship" in result


async def test_end_scene_clears_state(
    mock_context: MagicMock,
    dependencies: HaroldDependencies,
) -> None:
    """Verify that end_scene clears current_scene from dependencies.

    Args:
        mock_context: The mock RunContext fixture.
        dependencies: The wired dependency container fixture.
    """
    dependencies.current_scene = SceneState(
        setting="a pirate ship", suggestion="treasure"
    )

    result = await end_scene(
        mock_context, reason="reached a natural button"
    )

    assert dependencies.current_scene is None
    assert "Reason: reached a natural button" in result


async def test_end_scene_when_no_scene_active(
    mock_context: MagicMock,
    dependencies: HaroldDependencies,
) -> None:
    """Verify graceful handling when ending a scene with none active.

    Args:
        mock_context: The mock RunContext fixture.
        dependencies: The wired dependency container fixture.
    """
    assert dependencies.current_scene is None

    result = await end_scene(mock_context, reason="test")

    assert "No scene is currently active" in result
