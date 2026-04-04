"""Shared test configuration and fixtures for the Harold test suite.

Globally disables real LLM API requests via ALLOW_MODEL_REQUESTS=False,
ensuring all tests run against StubModel or FunctionModel. Provides
reusable fixtures for settings, memory backends, and the full
dependency container.
"""

from __future__ import annotations

import pydantic_ai
import pytest
from pydantic_ai.models.test import TestModel as StubModel

from harold.config import HaroldSettings
from harold.dependencies import HaroldDependencies
from harold.memory.backends.in_memory import (
    InMemoryLongTermMemory,
    InMemoryTrajectoryMemory,
)


def pytest_configure(config: pytest.Config) -> None:
    """Register custom pytest markers for selective test execution.

    Args:
        config: The pytest configuration object provided by the
            pytest plugin system.
    """
    config.addinivalue_line(
        "markers",
        "integration: marks tests requiring real API calls "
        "(deselect with '-m \"not integration\"')",
    )


pydantic_ai.settings.ALLOW_MODEL_REQUESTS = False


@pytest.fixture
def stub_model() -> StubModel:
    """Provide a StubModel that returns deterministic responses without API calls.

    Returns:
        A StubModel instance configured with default settings.
    """
    return StubModel()


@pytest.fixture
def settings() -> HaroldSettings:
    """Provide HaroldSettings with default values suitable for testing.

    Returns:
        A HaroldSettings instance using all default configuration values.
    """
    return HaroldSettings()


@pytest.fixture
def long_term_memory() -> InMemoryLongTermMemory:
    """Provide a fresh in-memory long-term memory backend with no stored data.

    Returns:
        An empty InMemoryLongTermMemory instance.
    """
    return InMemoryLongTermMemory()


@pytest.fixture
def trajectory_memory() -> InMemoryTrajectoryMemory:
    """Provide a fresh in-memory trajectory memory backend with no recorded scenes.

    Returns:
        An empty InMemoryTrajectoryMemory instance.
    """
    return InMemoryTrajectoryMemory()


@pytest.fixture
def dependencies(
    settings: HaroldSettings,
    long_term_memory: InMemoryLongTermMemory,
    trajectory_memory: InMemoryTrajectoryMemory,
) -> HaroldDependencies:
    """Provide a fully wired HaroldDependencies container with in-memory backends.

    Args:
        settings: Application configuration for the test run.
        long_term_memory: In-memory vector memory backend.
        trajectory_memory: In-memory trajectory graph backend.

    Returns:
        A HaroldDependencies instance ready for injection into the agent.
    """
    return HaroldDependencies(
        settings=settings,
        long_term_memory=long_term_memory,
        trajectory_memory=trajectory_memory,
    )
