"""Run Harold's improv quality evaluation suite.

Loads the eval dataset, runs each case through the scene partner
agent, and scores responses using both custom evaluators and LLM
judges. Results are printed to the terminal and traced via
OpenTelemetry when Phoenix is enabled.

Run with: ``uv run python evals/run_evals.py``
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from pydantic_evals import Dataset
from pydantic_evals.evaluators import LLMJudge

from evals.evaluators.improv import AcceptancePrinciple, AddsInformation
from harold.agents.scene_partner import scene_partner
from harold.config import HaroldSettings
from harold.dependencies import HaroldDependencies
from harold.memory.backends.in_memory import (
    InMemoryLongTermMemory,
    InMemoryTrajectoryMemory,
)
from harold.observability import setup_observability

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_PATH = Path(__file__).parent / "datasets" / "improv_quality.yaml"

YES_AND_RUBRIC = (
    "Does the response accept the scene partner's premise without "
    "denying or negating it, and add new information or context? "
    "Score 1 if yes, 0 if the response blocks or denies."
)

STRONG_CHOICES_RUBRIC = (
    "Does the response make strong, committed character choices? "
    "Does it name specific things, establish relationships, or "
    "commit to an emotional point of view rather than asking "
    "vague questions? Score 1 if yes, 0 if wishy-washy."
)

HEIGHTENING_RUBRIC = (
    "Does the response escalate the scene — raising the stakes, "
    "pushing the premise further, or intensifying the emotion? "
    "Score 1 if it heightens, 0 if it stays flat or retreats."
)


async def run_agent(inputs: str) -> str:
    """Run the scene partner agent and return the dialogue text.

    Args:
        inputs: The scene prompt to send to the agent.

    Returns:
        The agent's dialogue response as a plain string.
    """
    settings = HaroldSettings()
    dependencies = HaroldDependencies(
        settings=settings,
        long_term_memory=InMemoryLongTermMemory(),
        trajectory_memory=InMemoryTrajectoryMemory(),
    )
    result = await scene_partner.run(
        inputs,
        deps=dependencies,
        model=settings.llm_model,
    )
    return result.output.dialogue


def build_dataset() -> Dataset[str, str]:
    """Load eval cases from YAML and attach evaluators.

    Returns:
        A ``Dataset`` with improv quality cases and evaluators.
    """
    raw_dataset: Dataset[str, str] = Dataset.from_file(DATASET_PATH)

    return Dataset(
        name="harold_improv_quality",
        cases=raw_dataset.cases,
        evaluators=[
            AcceptancePrinciple(),
            AddsInformation(),
            LLMJudge(rubric=YES_AND_RUBRIC, include_input=True),
            LLMJudge(
                rubric=STRONG_CHOICES_RUBRIC, include_input=True
            ),
            LLMJudge(
                rubric=HEIGHTENING_RUBRIC, include_input=True
            ),
        ],
    )


async def main() -> None:
    """Load dataset, run evaluations, and print results."""
    settings = HaroldSettings()
    setup_observability(settings)

    logger.info("Loading eval dataset from %s", DATASET_PATH)
    dataset = build_dataset()
    logger.info("Running %d eval cases...", len(dataset.cases))

    report = await dataset.evaluate(run_agent)
    report.print()


if __name__ == "__main__":
    asyncio.run(main())
