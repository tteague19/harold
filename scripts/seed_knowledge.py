"""Seed the pgvector knowledge table with UCB improv principles.

Run once after setting up the database:

    uv run python scripts/seed_knowledge.py

Requires HAROLD_PG_DSN and OPENAI_API_KEY to be set.
"""

from __future__ import annotations

import asyncio
import logging

from harold.config import HaroldSettings
from harold.memory.backends.pgvector import PgVectorLongTermMemory
from harold.models.memory import KnowledgeCategory, KnowledgeEntry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UCB_PRINCIPLES: list[KnowledgeEntry] = [
    KnowledgeEntry(
        content=(
            "Yes, And: Always accept your scene partner's offer and "
            "build on it. Never deny what has been established. "
            "Agreement is the foundation of every scene."
        ),
        category=KnowledgeCategory.UCB_PRINCIPLE,
        source="UCB Comedy Improvisation Manual",
    ),
    KnowledgeEntry(
        content=(
            "Game of the Scene: Find the first unusual thing in "
            "the scene — the pattern or comic premise — and heighten "
            "it. Play the game by exploring it through three beats "
            "of escalation."
        ),
        category=KnowledgeCategory.UCB_PRINCIPLE,
        source="UCB Comedy Improvisation Manual",
    ),
    KnowledgeEntry(
        content=(
            "Heightening: Once the game is found, escalate it. Each "
            "beat should raise the stakes or push the premise further "
            "while staying within the internal logic of the scene."
        ),
        category=KnowledgeCategory.TECHNIQUE,
        source="UCB Comedy Improvisation Manual",
    ),
    KnowledgeEntry(
        content=(
            "Strong Choices: Name things. Establish the relationship, "
            "location, and activity early. Commit fully to your "
            "character. Avoid asking questions that put the burden "
            "on your scene partner."
        ),
        category=KnowledgeCategory.UCB_PRINCIPLE,
        source="UCB Comedy Improvisation Manual",
    ),
    KnowledgeEntry(
        content=(
            "Emotional Truth: Ground even the most absurd premises in "
            "genuine human emotion. The comedy comes from honest "
            "reactions to unusual circumstances."
        ),
        category=KnowledgeCategory.UCB_PRINCIPLE,
        source="UCB Comedy Improvisation Manual",
    ),
    KnowledgeEntry(
        content=(
            "Callbacks: Reference elements from earlier in the scene "
            "or from previous scenes. Callbacks reward the audience "
            "for paying attention and create a sense of cohesion."
        ),
        category=KnowledgeCategory.TECHNIQUE,
        source="UCB Comedy Improvisation Manual",
    ),
    KnowledgeEntry(
        content=(
            "If This Is True, What Else Is True: Once a premise is "
            "established, explore its logical implications. If your "
            "character is afraid of chairs, what does their apartment "
            "look like? How do they eat dinner?"
        ),
        category=KnowledgeCategory.TECHNIQUE,
        source="UCB Comedy Improvisation Manual",
    ),
    KnowledgeEntry(
        content=(
            "The Harold: A long-form improv format consisting of three "
            "separate scenes that are intercut and eventually connect "
            "through callbacks and thematic links. Begins with an "
            "opening inspired by a single audience suggestion."
        ),
        category=KnowledgeCategory.FORMAT,
        source="UCB Comedy Improvisation Manual",
    ),
    KnowledgeEntry(
        content=(
            "Group Mind: The ensemble's ability to think and react as "
            "one. Achieved through active listening, mirroring, and "
            "supporting scene partners' choices without hesitation."
        ),
        category=KnowledgeCategory.UCB_PRINCIPLE,
        source="UCB Comedy Improvisation Manual",
    ),
    KnowledgeEntry(
        content=(
            "Rest of the Scene: After the game has been played three "
            "times, the scene should find a natural ending. Look for "
            "a strong button — a final line or moment that punctuates "
            "the scene at its peak."
        ),
        category=KnowledgeCategory.TECHNIQUE,
        source="UCB Comedy Improvisation Manual",
    ),
]


async def seed() -> None:
    """Connect to pgvector and insert all UCB knowledge entries.

    Raises:
        ValueError: If ``HAROLD_PG_DSN`` is not configured.
    """
    settings = HaroldSettings()
    memory = await PgVectorLongTermMemory.create(settings)

    for entry in UCB_PRINCIPLES:
        await memory.store_knowledge(entry)
        logger.info("Seeded: %s", entry.content[:60])

    logger.info("Seeded %d knowledge entries", len(UCB_PRINCIPLES))


if __name__ == "__main__":
    asyncio.run(seed())
