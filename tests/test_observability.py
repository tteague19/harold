"""Tests for the observability bootstrap module.

Verifies that ``setup_observability`` correctly configures OpenTelemetry
tracing when enabled, and is a no-op when disabled.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from opentelemetry import trace

from harold.config import HaroldSettings
from harold.observability import setup_observability


def test_setup_observability_disabled() -> None:
    """Verify that setup_observability is a no-op when phoenix_enabled is False."""
    settings = HaroldSettings(phoenix_enabled=False)

    original_provider = trace.get_tracer_provider()
    setup_observability(settings)

    assert trace.get_tracer_provider() is original_provider


@patch("harold.observability.Agent")
def test_setup_observability_enabled(mock_agent: MagicMock) -> None:
    """Verify that setup_observability configures OTel when phoenix_enabled is True.

    Args:
        mock_agent: Mocked Pydantic AI Agent class to verify
            ``instrument_all`` is called.
    """
    settings = HaroldSettings(phoenix_enabled=True)

    setup_observability(settings)

    provider = trace.get_tracer_provider()
    assert hasattr(provider, "force_flush")
    mock_agent.instrument_all.assert_called_once()
