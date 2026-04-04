"""Tests for the FastAPI WebSocket interface.

Verifies the health endpoint, WebSocket session lifecycle, and
streaming message protocol. Uses StubModel to avoid real API calls
and FastAPI's TestClient for HTTP/WebSocket testing.
"""

from __future__ import annotations

import json

from pydantic_ai.models.test import TestModel as StubModel
from starlette.testclient import TestClient

from harold.agents.scene_partner import scene_partner
from harold.interfaces.api import (
    APP_VERSION,
    HealthResponse,
    ServerMessageType,
    app,
)

HEALTH_ENDPOINT = "/health"
WEBSOCKET_PATH = "/ws/test-session"
EXPECTED_HEALTHY_STATUS = "healthy"


def test_health_endpoint() -> None:
    """Verify that GET /health returns 200 with status and version.

    Asserts the response matches the ``HealthResponse`` model with
    the expected status and application version.
    """
    with TestClient(app) as client:
        response = client.get(HEALTH_ENDPOINT)

    assert response.status_code == 200
    body = HealthResponse(**response.json())
    assert body.status == EXPECTED_HEALTHY_STATUS
    assert body.version == APP_VERSION


def test_websocket_session_receives_response() -> None:
    """Verify that a WebSocket session streams chunks and a final response.

    Sends a single user message and expects at least one stream chunk
    followed by a response message containing the structured output.
    """
    with (
        scene_partner.override(model=StubModel()),
        TestClient(app) as client,
        client.websocket_connect(WEBSOCKET_PATH) as ws,
    ):
        ws.send_text(
            json.dumps(
                {"type": "message", "content": "Hello!"}
            )
        )

        messages = []
        while True:
            data = ws.receive_json()
            messages.append(data)
            if data["type"] == ServerMessageType.RESPONSE:
                break

    response_messages = [
        m
        for m in messages
        if m["type"] == ServerMessageType.RESPONSE
    ]

    assert len(response_messages) == 1
    assert "dialogue" in response_messages[0]
    assert "emotional_tone" in response_messages[0]


def test_websocket_empty_message_returns_error() -> None:
    """Verify that an empty message content returns an error.

    Sends a message with empty content and expects an error response
    rather than an agent invocation.
    """
    with (
        scene_partner.override(model=StubModel()),
        TestClient(app) as client,
        client.websocket_connect(WEBSOCKET_PATH) as ws,
    ):
        ws.send_text(
            json.dumps({"type": "message", "content": ""})
        )
        data = ws.receive_json()

    assert data["type"] == ServerMessageType.ERROR
    assert "Empty" in data["detail"]
