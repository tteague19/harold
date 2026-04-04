"""Pydantic models for coaching feedback.

Defines the structured output produced by the coaching agent when
analyzing a user's improv patterns and providing UCB-grounded advice.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

MIN_FEEDBACK_ITEMS = 1


class CoachingFeedback(BaseModel):
    """Structured coaching feedback based on the user's improv history.

    Attributes:
        strengths: Specific improv skills the user demonstrates well.
        growth_areas: Areas where the user could improve, with
            concrete observations.
        specific_tips: Actionable advice grounded in UCB principles.
        technique_suggestion: A single technique the user should try
            in their next scene.
    """

    strengths: list[str] = Field(
        min_length=MIN_FEEDBACK_ITEMS,
        description="Specific improv skills the user demonstrates well",
    )
    growth_areas: list[str] = Field(
        min_length=MIN_FEEDBACK_ITEMS,
        description="Areas where the user could improve",
    )
    specific_tips: list[str] = Field(
        min_length=MIN_FEEDBACK_ITEMS,
        description="Actionable advice grounded in UCB principles",
    )
    technique_suggestion: str = Field(
        min_length=1,
        description=(
            "A single technique to try in the next scene"
        ),
    )
