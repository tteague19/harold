"""Custom evaluators for improv quality assessment.

Provides domain-specific evaluators grounded in UCB Comedy
Improvisation Manual principles. These are used alongside
``LLMJudge`` instances in the eval dataset to measure Harold's
performance across multiple improv dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic_evals.evaluators import Evaluator, EvaluatorContext

NEGATION_PATTERNS = [
    "no,",
    "no.",
    "no!",
    "that's not",
    "that isn't",
    "but actually",
    "i don't think",
    "you can't",
    "we can't",
    "that doesn't",
    "wait, no",
]

MINIMUM_NOVEL_WORD_RATIO = 0.3


@dataclass
class AcceptancePrinciple(Evaluator[str, str]):
    """Check that the response follows the yes-and principle.

    Scans the output for negation patterns that would indicate
    the agent is denying or blocking the scene partner's offer.
    Returns ``True`` if no negation patterns are found.
    """

    def evaluate(self, ctx: EvaluatorContext[str, str]) -> bool:
        """Evaluate whether the response avoids negation.

        Args:
            ctx: The evaluation context containing input, output,
                and metadata.

        Returns:
            ``True`` if the response contains no negation patterns.
        """
        output_lower = ctx.output.lower()
        return not any(
            pattern in output_lower for pattern in NEGATION_PATTERNS
        )


@dataclass
class AddsInformation(Evaluator[str, str]):
    """Score how much new content the response introduces.

    Computes the ratio of novel words (words in the output but not
    in the input) to total output words. A higher ratio indicates
    the agent is adding more new information to the scene.
    """

    def evaluate(self, ctx: EvaluatorContext[str, str]) -> float:
        """Evaluate the novelty ratio of the response.

        Args:
            ctx: The evaluation context containing input, output,
                and metadata.

        Returns:
            A float between 0.0 and 1.0 representing the proportion
            of novel words in the response.
        """
        input_words = set(ctx.inputs.lower().split())
        output_words = ctx.output.lower().split()
        if not output_words:
            return 0.0
        novel_count = sum(
            1 for word in output_words if word not in input_words
        )
        return novel_count / len(output_words)
