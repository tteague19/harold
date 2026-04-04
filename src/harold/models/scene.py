"""Pydantic models representing improv scenes, turns, and responses.

These models are used throughout the agent, memory, and interface layers
to enforce structured data at every boundary.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field

MIN_DIALOGUE_LENGTH = 1
MAX_STAGE_DIRECTION_LENGTH = 500


class Speaker(StrEnum):
    """Identifies who delivered a beat in a scene.

    Attributes:
        USER: The human performer.
        HAROLD: The AI scene partner.
    """

    USER = "user"
    HAROLD = "harold"


class EmotionalTone(StrEnum):
    """The emotional quality of a scene beat.

    Used to tag each response so trajectory memory can track
    emotional arcs across scenes.

    Attributes:
        NEUTRAL: No strong emotional coloring.
        JOYFUL: Happiness, delight, or celebration.
        ANXIOUS: Worry, nervousness, or dread.
        ANGRY: Frustration, rage, or indignation.
        CONFUSED: Bewilderment or disorientation.
        EXCITED: High energy enthusiasm.
        MELANCHOLY: Sadness or wistfulness.
        PLAYFUL: Light-hearted mischief.
        SINCERE: Earnest emotional honesty.
        ABSURD: Surreal or nonsensical energy.
    """

    NEUTRAL = "neutral"
    JOYFUL = "joyful"
    ANXIOUS = "anxious"
    ANGRY = "angry"
    CONFUSED = "confused"
    EXCITED = "excited"
    MELANCHOLY = "melancholy"
    PLAYFUL = "playful"
    SINCERE = "sincere"
    ABSURD = "absurd"


class SceneResponse(BaseModel):
    """The agent's structured response within an improv scene.

    Attributes:
        dialogue: The character's spoken dialogue or narration.
        stage_direction: Optional physical action or stage business.
        emotional_tone: The emotional quality of this beat.
        callback_used: Reference to an earlier scene element, if any.
    """

    dialogue: str = Field(
        min_length=MIN_DIALOGUE_LENGTH,
        description="The character's dialogue or narration",
    )
    stage_direction: str | None = Field(
        default=None,
        max_length=MAX_STAGE_DIRECTION_LENGTH,
        description="Optional physical action or stage direction",
    )
    emotional_tone: EmotionalTone = Field(
        default=EmotionalTone.NEUTRAL,
        description="The emotional quality of this beat",
    )
    callback_used: str | None = Field(
        default=None,
        description="Reference to an earlier scene element",
    )


class Turn(BaseModel):
    """A single beat in a scene.

    Attributes:
        speaker: Who delivered this beat.
        content: The spoken or narrated content.
        timestamp: When this beat occurred.
    """

    speaker: Speaker = Field(description="Who delivered this beat")
    content: str = Field(
        min_length=MIN_DIALOGUE_LENGTH,
        description="The spoken or narrated content",
    )
    timestamp: datetime = Field(default_factory=datetime.now)


class SceneState(BaseModel):
    """Mutable state tracking the current in-progress scene.

    Attributes:
        scene_id: Unique identifier for this scene.
        setting: The location or context of the scene.
        suggestion: The audience suggestion that inspired the scene.
        turns: Ordered list of beats played so far.
        started_at: When the scene began.
    """

    scene_id: str = Field(default_factory=lambda: str(uuid4()))
    setting: str = Field(
        min_length=1,
        description="The location or context of the scene",
    )
    suggestion: str = Field(
        min_length=1,
        description="The audience suggestion that inspired the scene",
    )
    turns: list[Turn] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=datetime.now)


class SceneSummary(BaseModel):
    """Persisted after a scene ends for long-term memory retrieval.

    Attributes:
        scene_id: ID of the scene this summarizes.
        setting: The location or context of the scene.
        suggestion: The audience suggestion that inspired the scene.
        summary: Brief narrative summary of what happened.
        key_moments: Notable beats or discoveries in the scene.
        techniques_used: Improv techniques observed during the scene.
        duration_turns: Total number of turns in the scene.
    """

    scene_id: str = Field(description="ID of the scene this summarizes")
    setting: str = Field(min_length=1)
    suggestion: str = Field(min_length=1)
    summary: str = Field(
        min_length=1,
        description="Brief narrative summary of the scene",
    )
    key_moments: list[str] = Field(
        min_length=1,
        description="Notable beats or discoveries in the scene",
    )
    techniques_used: list[str] = Field(
        default_factory=list,
        description="Improv techniques observed during the scene",
    )
    duration_turns: int = Field(
        ge=1,
        description="Total number of turns in the scene",
    )
