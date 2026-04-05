"""Tests for the pattern analyzer agent.

Verifies that the pattern analyzer produces valid ImprovWorkflow
output when run with StubModel against trajectory data.
"""

from __future__ import annotations

from pydantic_ai.models.test import TestModel as StubModel

from harold.agents.pattern_analyzer import pattern_analyzer
from harold.dependencies import HaroldDependencies
from harold.models.scene import SceneSummary

SEED_SCENE_A_ID = "analyzer-test-a"
SEED_SCENE_B_ID = "analyzer-test-b"
SEED_SETTING = "a coffee shop"
SEED_SUGGESTION = "coffee"
SEED_SUMMARY_A = "Two baristas compete to make the perfect latte"
SEED_SUMMARY_B = "A customer orders an impossible drink"
SEED_KEY_MOMENT = "the milk exploded"
SEED_TECHNIQUES = ["yes-and", "heightening"]
SEED_DURATION = 4


async def test_pattern_analyzer_returns_workflows(
    dependencies: HaroldDependencies,
    stub_model: StubModel,
) -> None:
    """Verify the analyzer produces a list of ImprovWorkflow with StubModel.

    Args:
        dependencies: The wired dependency container fixture.
        stub_model: The deterministic stub model fixture.
    """
    with pattern_analyzer.override(model=stub_model):
        result = await pattern_analyzer.run(
            "Analyze my scene history and extract patterns.",
            deps=dependencies,
            model=stub_model,
        )

    assert isinstance(result.output, list)


async def test_pattern_analyzer_with_scene_data(
    dependencies: HaroldDependencies,
    stub_model: StubModel,
) -> None:
    """Verify the analyzer runs when trajectory memory has data.

    Args:
        dependencies: The wired dependency container fixture.
        stub_model: The deterministic stub model fixture.
    """
    for scene_id, summary in [
        (SEED_SCENE_A_ID, SEED_SUMMARY_A),
        (SEED_SCENE_B_ID, SEED_SUMMARY_B),
    ]:
        scene = SceneSummary(
            scene_id=scene_id,
            setting=SEED_SETTING,
            suggestion=SEED_SUGGESTION,
            summary=summary,
            key_moments=[SEED_KEY_MOMENT],
            techniques_used=SEED_TECHNIQUES,
            duration_turns=SEED_DURATION,
        )
        await dependencies.trajectory_memory.record_scene(scene)

    with pattern_analyzer.override(model=stub_model):
        result = await pattern_analyzer.run(
            "Analyze my scene history and extract patterns.",
            deps=dependencies,
            model=stub_model,
        )

    assert isinstance(result.output, list)
