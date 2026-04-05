"""Pydantic models for multi-character scene support.

Defines the ``Character`` model representing an AI-controlled
improv character with distinct personality traits and speaking style.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Character(BaseModel):
    """An AI-controlled improv character with distinct traits.

    Attributes:
        name: The character's name as it appears in dialogue.
        personality: A brief description of the character's
            personality and behavioral tendencies.
        speaking_style: How the character talks — accent, vocabulary,
            pacing, or other speech patterns.
    """

    name: str = Field(
        min_length=1,
        description="The character's name",
    )
    personality: str = Field(
        min_length=1,
        description="Brief personality description",
    )
    speaking_style: str = Field(
        min_length=1,
        description="How the character talks",
    )
