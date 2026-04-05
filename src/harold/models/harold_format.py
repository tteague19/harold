"""Pydantic models for the Harold long-form improv format.

The Harold format consists of an audience suggestion, an opening
exploration, and three connected scenes with callbacks and thematic
links between them.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from harold.models.types import SceneNumber

DEFAULT_SCENE_COUNT: SceneNumber = 3


class HaroldShow(BaseModel):
    """State tracking for a Harold long-form improv performance.

    Attributes:
        suggestion: The audience suggestion that inspires the show.
        opening_summary: A summary of the opening exploration that
            establishes themes and ideas for the scenes.
        scene_summaries: Summaries of completed scenes, accumulated
            as the show progresses.
        total_scenes: The number of scenes in this Harold.
    """

    suggestion: str = Field(
        min_length=1,
        description="The audience suggestion inspiring the show",
    )
    opening_summary: str = Field(
        default="",
        description="Summary of the opening exploration",
    )
    scene_summaries: list[str] = Field(
        default_factory=list,
        description="Summaries of completed scenes so far",
    )
    total_scenes: SceneNumber = Field(
        default=DEFAULT_SCENE_COUNT,
        description="Number of scenes in this Harold",
    )
