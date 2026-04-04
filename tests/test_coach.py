"""Tests for the coaching agent.

Verifies that the coaching agent produces valid CoachingFeedback
output when run with StubModel. Uses the conftest dependencies
fixture which provides in-memory trajectory memory.
"""

from __future__ import annotations

from pydantic_ai.models.test import TestModel as StubModel

from harold.agents.coach import coach
from harold.dependencies import HaroldDependencies
from harold.models.coaching import CoachingFeedback
from harold.models.scene import SceneSummary

SEED_SCENE_ID = "coach-test-scene"
SEED_SETTING = "a dentist office"
SEED_SUGGESTION = "teeth"
SEED_SUMMARY = "Two dentists argue about the best flossing technique"
SEED_KEY_MOMENT = "the floss started singing"
SEED_TECHNIQUES = ["yes-and", "heightening"]
SEED_DURATION = 4


async def test_coach_returns_coaching_feedback(
    dependencies: HaroldDependencies,
    stub_model: StubModel,
) -> None:
    """Verify the coach produces valid CoachingFeedback with StubModel.

    Args:
        dependencies: The wired dependency container fixture.
        stub_model: The deterministic stub model fixture.
    """
    with coach.override(model=stub_model):
        result = await coach.run(
            "Give me coaching feedback.",
            deps=dependencies,
            model=stub_model,
        )

    assert isinstance(result.output, CoachingFeedback)
    assert len(result.output.strengths) >= 1
    assert len(result.output.growth_areas) >= 1
    assert len(result.output.specific_tips) >= 1
    assert len(result.output.technique_suggestion) >= 1


async def test_coach_with_scene_history(
    dependencies: HaroldDependencies,
    stub_model: StubModel,
) -> None:
    """Verify the coach runs successfully when trajectory memory has data.

    Args:
        dependencies: The wired dependency container fixture.
        stub_model: The deterministic stub model fixture.
    """
    scene = SceneSummary(
        scene_id=SEED_SCENE_ID,
        setting=SEED_SETTING,
        suggestion=SEED_SUGGESTION,
        summary=SEED_SUMMARY,
        key_moments=[SEED_KEY_MOMENT],
        techniques_used=SEED_TECHNIQUES,
        duration_turns=SEED_DURATION,
    )
    await dependencies.trajectory_memory.record_scene(scene)

    with coach.override(model=stub_model):
        result = await coach.run(
            "Give me coaching feedback.",
            deps=dependencies,
            model=stub_model,
        )

    assert isinstance(result.output, CoachingFeedback)
