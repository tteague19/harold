"""Dependency container for Pydantic AI agent injection.

The ``HaroldDependencies`` dataclass bundles all services the agent needs
and is passed via ``RunContext`` to every tool call.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic_ai.messages import ModelMessage

from harold.config import HaroldSettings
from harold.memory.base import LongTermMemory, TrajectoryMemory
from harold.models.character import Character
from harold.models.harold_format import HaroldShow
from harold.models.scene import SceneState


@dataclass
class HaroldDependencies:
    """Dependency container injected into the Pydantic AI agent via RunContext.

    Attributes:
        settings: Application configuration.
        long_term_memory: Backend for vector-searchable persistent memory.
        trajectory_memory: Backend for graph-based interaction traces.
        message_history: Accumulated Pydantic AI messages for conversation
            continuity across agent runs.
        current_scene: The in-progress scene state, or ``None`` if no
            scene is active.
        active_character: The character Harold is currently playing,
            or ``None`` for default behavior.
        characters: Available characters for multi-character scenes.
        harold_show: The active Harold format show, or ``None``
            if not running a Harold.
    """

    settings: HaroldSettings
    long_term_memory: LongTermMemory
    trajectory_memory: TrajectoryMemory
    message_history: list[ModelMessage] = field(default_factory=list)
    current_scene: SceneState | None = None
    active_character: Character | None = None
    characters: list[Character] = field(default_factory=list)
    harold_show: HaroldShow | None = None
