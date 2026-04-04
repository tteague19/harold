"""Tests for the scene partner agent.

Verifies that the agent produces valid SceneResponse output,
accumulates conversation history, and injects scene context
into the system prompt. All tests use StubModel to avoid real
API calls.
"""

from __future__ import annotations

from pydantic_ai.models.test import TestModel as StubModel

from harold.agents.scene_partner import scene_partner
from harold.dependencies import HaroldDependencies
from harold.models.scene import SceneResponse, SceneState


async def test_agent_returns_scene_response(
    dependencies: HaroldDependencies,
    stub_model: StubModel,
) -> None:
    """Verify the agent produces a valid SceneResponse when run with StubModel.

    Args:
        dependencies: The wired dependency container fixture.
        stub_model: The deterministic stub model fixture.
    """
    with scene_partner.override(model=stub_model):
        result = await scene_partner.run(
            "I'm a barista and this is your first day of training.",
            deps=dependencies,
            model=stub_model,
        )

    assert isinstance(result.output, SceneResponse)
    assert len(result.output.dialogue) > 0


async def test_conversation_history_accumulates(
    dependencies: HaroldDependencies,
    stub_model: StubModel,
) -> None:
    """Verify that new_messages grows with each successive agent run.

    Args:
        dependencies: The wired dependency container fixture.
        stub_model: The deterministic stub model fixture.
    """
    with scene_partner.override(model=stub_model):
        first_result = await scene_partner.run(
            "Welcome to the moon bakery!",
            deps=dependencies,
            model=stub_model,
        )
        first_messages = first_result.new_messages()

        second_result = await scene_partner.run(
            "The croissants keep floating away.",
            deps=dependencies,
            message_history=first_messages,
            model=stub_model,
        )
        all_messages = first_messages + second_result.new_messages()

    assert len(all_messages) > len(first_messages)


async def test_agent_with_active_scene_context(
    dependencies: HaroldDependencies,
    stub_model: StubModel,
) -> None:
    """Verify the system prompt includes scene details when a scene is active.

    Args:
        dependencies: The wired dependency container fixture.
        stub_model: The deterministic stub model fixture.
    """
    dependencies.current_scene = SceneState(
        setting="a haunted library",
        suggestion="books",
    )

    with scene_partner.override(model=stub_model):
        result = await scene_partner.run(
            "Did you hear that noise?",
            deps=dependencies,
            model=stub_model,
        )

    assert isinstance(result.output, SceneResponse)
