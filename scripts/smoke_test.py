"""Programmatic smoke test for Harold's core agent paths.

Exercises the scene partner and coaching agents with scripted prompts
and verifies structured output without interactive input. Requires
``ANTHROPIC_API_KEY`` but no Docker infrastructure.

Run with: ``uv run python scripts/smoke_test.py``
"""

from __future__ import annotations

import asyncio
import logging
import sys

from harold.agents.coach import coach
from harold.agents.scene_partner import scene_partner
from harold.config import HaroldSettings
from harold.dependencies import HaroldDependencies
from harold.memory.backends.in_memory import (
    InMemoryLongTermMemory,
    InMemoryTrajectoryMemory,
)
from harold.models.coaching import CoachingFeedback
from harold.models.scene import SceneResponse, SceneSummary

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SCENE_PROMPTS = [
    "I'm a barista and this is your first day of training. Welcome!",
    "Oh no, the espresso machine is making that sound again.",
    "Quick, grab the fire extinguisher — the pastry case is glowing!",
]

SEED_SCENE_ID = "smoke-test-scene"
SEED_SETTING = "a coffee shop"
SEED_SUGGESTION = "barista"
SEED_SUMMARY = "Two baristas deal with a malfunctioning espresso machine"
SEED_KEY_MOMENT = "the pastry case started glowing"
SEED_TECHNIQUES = ["yes-and", "heightening"]
SEED_DURATION = 3


async def smoke_test() -> bool:
    """Run the full smoke test and return whether it passed.

    Returns:
        ``True`` if all checks passed, ``False`` otherwise.
    """
    settings = HaroldSettings()
    dependencies = HaroldDependencies(
        settings=settings,
        long_term_memory=InMemoryLongTermMemory(),
        trajectory_memory=InMemoryTrajectoryMemory(),
    )

    logger.info("Testing scene partner agent...")
    message_history = []
    for prompt in SCENE_PROMPTS:
        result = await scene_partner.run(
            prompt,
            deps=dependencies,
            message_history=message_history,
            model=settings.llm_model,
        )
        message_history.extend(result.new_messages())

        if not isinstance(result.output, SceneResponse):
            logger.error("Expected SceneResponse, got %s", type(result.output))
            return False

        logger.info(
            "Harold: %s [%s]",
            result.output.dialogue,
            result.output.emotional_tone,
        )

    logger.info("Scene partner: PASSED (%d turns)", len(SCENE_PROMPTS))

    seed_scene = SceneSummary(
        scene_id=SEED_SCENE_ID,
        setting=SEED_SETTING,
        suggestion=SEED_SUGGESTION,
        summary=SEED_SUMMARY,
        key_moments=[SEED_KEY_MOMENT],
        techniques_used=SEED_TECHNIQUES,
        duration_turns=SEED_DURATION,
    )
    await dependencies.trajectory_memory.record_scene(seed_scene)

    logger.info("Testing coaching agent...")
    coach_result = await coach.run(
        "Review my improv history and give me coaching feedback.",
        deps=dependencies,
        model=settings.llm_model,
    )

    if not isinstance(coach_result.output, CoachingFeedback):
        logger.error(
            "Expected CoachingFeedback, got %s",
            type(coach_result.output),
        )
        return False

    feedback = coach_result.output
    logger.info("Strengths: %s", feedback.strengths)
    logger.info("Growth areas: %s", feedback.growth_areas)
    logger.info("Suggestion: %s", feedback.technique_suggestion)
    logger.info("Coaching agent: PASSED")

    return True


if __name__ == "__main__":
    passed = asyncio.run(smoke_test())
    sys.exit(0 if passed else 1)
