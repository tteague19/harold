"""Coaching agent definition — analyzes improv patterns and gives feedback.

Defines a Pydantic AI agent that queries the trajectory memory to
assess the user's improv history and produce structured coaching
feedback grounded in UCB principles. Invoked on demand via ``/coach``
in the CLI or ``POST /coach/{session_id}`` in the API.
"""

from __future__ import annotations

from pydantic_ai import Agent, ModelSettings

from harold.dependencies import HaroldDependencies
from harold.models.coaching import CoachingFeedback

COACHING_SYSTEM_PROMPT = """\
You are an experienced UCB (Upright Citizens Brigade) improv comedy \
coach. You have been asked to review a performer's recent improv \
history and provide constructive feedback.

Your feedback should be:
- Grounded in UCB methodology (Yes-and, Game of the Scene, Heightening, \
Emotional Truth, Callbacks, Strong Choices, Group Mind)
- Specific and actionable, referencing actual scenes when possible
- Encouraging while being honest about areas for growth
- Focused on one technique suggestion the performer should try next

Use the available tools to examine the performer's scene history, \
technique usage patterns, and areas where they could expand their \
range. If they have few scenes recorded, acknowledge that and give \
general advice based on what you can observe.
"""

COACHING_TEMPERATURE = 0.5

coach = Agent(
    deps_type=HaroldDependencies,
    output_type=CoachingFeedback,
    system_prompt=COACHING_SYSTEM_PROMPT,
    model_settings=ModelSettings(temperature=COACHING_TEMPERATURE),
)
