"""Reusable annotated types and module-level constants for domain validation.

Defines constrained type aliases used across the memory and tool layers
to enforce invariants (non-empty queries, positive limits) at the type level.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

SearchQuery = Annotated[
    str,
    Field(min_length=1, description="Non-empty search query"),
]
"""A non-empty string used as a search query across memory backends."""

SearchLimit = Annotated[
    int,
    Field(ge=1, le=100, description="Maximum number of results to return"),
]
"""A positive integer capping the number of search results returned."""

DEFAULT_SEARCH_LIMIT = 5
DEFAULT_SCENE_RECALL_LIMIT = 3
DEFAULT_KNOWLEDGE_RECALL_LIMIT = 2
