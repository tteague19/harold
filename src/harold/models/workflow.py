"""Pydantic model for improv workflow templates.

Represents reusable patterns discovered from trajectory analysis
that can be injected into the scene partner's context at runtime.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

MIN_TECHNIQUE_SEQUENCE_LENGTH = 1
MIN_SUCCESS_COUNT = 1


class ImprovWorkflow(BaseModel):
    """A reusable improv workflow extracted from trajectory data.

    Attributes:
        name: Short identifier for this workflow pattern.
        description: What this workflow achieves and why it works.
        scene_type: The category of scene this workflow applies to.
        technique_sequence: Ordered list of techniques that form
            this workflow.
        trigger_description: When to apply this workflow in a scene.
        example_summary: A concrete example scene from trajectory
            data that demonstrates this workflow.
        success_count: Number of scenes where this pattern was
            observed successfully.
    """

    name: str = Field(
        min_length=1,
        description="Short identifier for this workflow pattern",
    )
    description: str = Field(
        min_length=1,
        description="What this workflow achieves and why it works",
    )
    scene_type: str = Field(
        min_length=1,
        description="Category of scene this workflow applies to",
    )
    technique_sequence: list[str] = Field(
        min_length=MIN_TECHNIQUE_SEQUENCE_LENGTH,
        description="Ordered techniques that form this workflow",
    )
    trigger_description: str = Field(
        min_length=1,
        description="When to apply this workflow in a scene",
    )
    example_summary: str = Field(
        min_length=1,
        description="A concrete example from trajectory data",
    )
    success_count: int = Field(
        ge=MIN_SUCCESS_COUNT,
        description="Number of scenes using this pattern",
    )
