"""Pydantic models for improv knowledge stored in long-term memory.

These models represent the structured knowledge entries that the agent
can retrieve via tool calls to inform its improv choices.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class KnowledgeCategory(StrEnum):
    """Classification of improv knowledge entries.

    Attributes:
        UCB_PRINCIPLE: A core UCB improvisation principle.
        TECHNIQUE: A specific performance technique.
        GAME: A recognized scene game or pattern.
        FORMAT: A long-form improv format structure.
        EXERCISE: A practice exercise or warm-up.
    """

    UCB_PRINCIPLE = "ucb_principle"
    TECHNIQUE = "technique"
    GAME = "game"
    FORMAT = "format"
    EXERCISE = "exercise"


class KnowledgeEntry(BaseModel):
    """A piece of improv knowledge stored in long-term memory.

    Attributes:
        content: The knowledge content text.
        category: Classification of this knowledge entry.
        source: Origin of this knowledge, e.g. book title or
            instructor name.
    """

    content: str = Field(
        min_length=1,
        description="The knowledge content",
    )
    category: KnowledgeCategory = Field(
        description="Classification of this knowledge entry",
    )
    source: str | None = Field(
        default=None,
        description="Origin of this knowledge, e.g. book or instructor",
    )
