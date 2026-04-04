"""FastAPI WebSocket interface for streaming improv sessions.

Exposes Harold as an HTTP API with a WebSocket endpoint for real-time
scene interaction. Uses ``agent.run_stream()`` to deliver partial text
chunks as they arrive from the LLM, followed by the complete structured
``SceneResponse``.

Start with: ``uv run harold-api`` or
``uv run uvicorn harold.interfaces.api:app --reload``
"""

from __future__ import annotations

import contextlib
import json
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from enum import StrEnum

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydantic_ai.messages import ModelMessage

from harold.agents.coach import coach
from harold.agents.scene_partner import scene_partner
from harold.bootstrap import build_dependencies
from harold.config import HaroldSettings
from harold.dependencies import HaroldDependencies
from harold.models.coaching import CoachingFeedback
from harold.observability import setup_observability

logger = logging.getLogger(__name__)

DEFAULT_API_HOST = "0.0.0.0"
DEFAULT_API_PORT = 8000
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:8080",
]
APP_VERSION = "0.1.0"


class ServerMessageType(StrEnum):
    """Types of messages sent from the server to WebSocket clients.

    Attributes:
        STREAM: A partial text chunk from the LLM response.
        RESPONSE: The complete structured SceneResponse.
        ERROR: An error message.
    """

    STREAM = "stream"
    RESPONSE = "response"
    ERROR = "error"


class ClientMessageType(StrEnum):
    """Types of messages expected from WebSocket clients.

    Attributes:
        MESSAGE: A user message to send to the scene partner.
    """

    MESSAGE = "message"


class HealthResponse(BaseModel):
    """Response model for the health check endpoint.

    Attributes:
        status: The current health status of the API.
        version: The application version string.
    """

    status: str = Field(description="Current health status")
    version: str = Field(description="Application version string")


def _parse_client_message(raw: str) -> str | None:
    """Extract the content string from a raw WebSocket message.

    Args:
        raw: The raw JSON string received from the client.

    Returns:
        The message content, or ``None`` if the content is empty
        or missing.
    """
    data = json.loads(raw)
    content = data.get("content", "")
    return content if content else None


async def _send_error(websocket: WebSocket, detail: str) -> None:
    """Send an error message to the WebSocket client.

    Args:
        websocket: The WebSocket connection.
        detail: A human-readable error description.
    """
    await websocket.send_json(
        {"type": ServerMessageType.ERROR, "detail": detail}
    )


async def _stream_agent_response(
    websocket: WebSocket,
    content: str,
    settings: HaroldSettings,
    dependencies: HaroldDependencies,
    message_history: list[ModelMessage],
) -> None:
    """Run the agent in streaming mode and send results to the client.

    Streams partial ``SceneResponse`` objects as they build up,
    sending the dialogue text incrementally. Once the stream
    completes, sends the final structured response and appends
    new messages to the conversation history.

    Args:
        websocket: The WebSocket connection to stream to.
        content: The user's message text.
        settings: Application configuration for model selection.
        dependencies: The wired dependency container for the agent.
        message_history: The session's accumulated conversation
            history, mutated in place with new messages.
    """
    previous_dialogue = ""
    async with scene_partner.run_stream(
        content,
        deps=dependencies,
        message_history=message_history,
        model=settings.llm_model,
    ) as stream:
        async for partial in stream.stream_output(debounce_by=None):
            current_dialogue = partial.dialogue
            if not current_dialogue or current_dialogue == previous_dialogue:
                continue
            delta = current_dialogue[len(previous_dialogue):]
            await websocket.send_json(
                {
                    "type": ServerMessageType.STREAM,
                    "content": delta,
                }
            )
            previous_dialogue = current_dialogue

    output = await stream.get_output()
    message_history.extend(stream.new_messages())

    await websocket.send_json(
        {
            "type": ServerMessageType.RESPONSE,
            "dialogue": output.dialogue,
            "stage_direction": output.stage_direction,
            "emotional_tone": output.emotional_tone.value,
            "callback_used": output.callback_used,
        }
    )


async def _handle_session_loop(
    websocket: WebSocket,
    settings: HaroldSettings,
    dependencies: HaroldDependencies,
    message_history: list[ModelMessage],
) -> None:
    """Run the receive-process-respond loop for a WebSocket session.

    Loops until the client disconnects, parsing each incoming message
    and streaming the agent's response back.

    Args:
        websocket: The WebSocket connection.
        settings: Application configuration for model selection.
        dependencies: The wired dependency container for the agent.
        message_history: The session's accumulated conversation history.
    """
    while True:
        raw = await websocket.receive_text()
        content = _parse_client_message(raw)

        if content is None:
            await _send_error(websocket, "Empty message content")
            continue

        await _stream_agent_response(
            websocket,
            content,
            settings,
            dependencies,
            message_history,
        )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize shared application state at startup and tear down on shutdown.

    Creates settings, configures observability, and builds the
    dependency container. Stores them on ``app.state`` for access
    in endpoint handlers.

    Args:
        app: The FastAPI application instance.

    Yields:
        Control to the running application.
    """
    settings = HaroldSettings()
    setup_observability(settings)
    dependencies = await build_dependencies(settings)

    app.state.settings = settings
    app.state.dependencies = dependencies
    sessions: dict[str, list[ModelMessage]] = {}
    app.state.sessions = sessions

    logger.info("Harold API started")
    yield
    logger.info("Harold API shutting down")


app = FastAPI(
    title="Harold",
    description="AI improv comedy partner and coach",
    version=APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return the API health status and version.

    Returns:
        A ``HealthResponse`` with status and version.
    """
    return HealthResponse(status="healthy", version=APP_VERSION)


@app.post("/coach/{session_id}", response_model=CoachingFeedback)
async def request_coaching(session_id: str) -> CoachingFeedback:
    """Run the coaching agent and return structured feedback.

    Analyzes the user's improv history via the trajectory memory
    backend and returns UCB-grounded coaching advice.

    Args:
        session_id: The session identifier for context.

    Returns:
        A ``CoachingFeedback`` with strengths, growth areas, tips,
        and a technique suggestion.
    """
    settings: HaroldSettings = app.state.settings
    dependencies: HaroldDependencies = app.state.dependencies

    result = await coach.run(
        "Review my improv history and give me coaching feedback.",
        deps=dependencies,
        model=settings.llm_model,
    )
    return result.output


@app.websocket("/ws/{session_id}")
async def websocket_session(
    websocket: WebSocket, session_id: str
) -> None:
    """Handle a streaming improv session over WebSocket.

    Maintains per-session conversation history. For each user message,
    streams partial text chunks from the LLM, then sends the complete
    structured SceneResponse.

    Args:
        websocket: The WebSocket connection.
        session_id: Unique identifier for this conversation session.
    """
    await websocket.accept()

    settings: HaroldSettings = websocket.app.state.settings
    dependencies: HaroldDependencies = (
        websocket.app.state.dependencies
    )
    sessions: dict[str, list[ModelMessage]] = (
        websocket.app.state.sessions
    )

    if session_id not in sessions:
        sessions[session_id] = []

    message_history = sessions[session_id]

    try:
        await _handle_session_loop(
            websocket, settings, dependencies, message_history
        )
    except WebSocketDisconnect:
        logger.info("Session %s disconnected", session_id)
    except Exception:
        logger.exception("Error in session %s", session_id)
        with contextlib.suppress(Exception):
            await _send_error(websocket, "Internal server error")


def run() -> None:
    """Start the API server via uvicorn.

    Used as the ``harold-api`` console script entry point.
    """
    import uvicorn

    uvicorn.run(
        "harold.interfaces.api:app",
        host=DEFAULT_API_HOST,
        port=DEFAULT_API_PORT,
        reload=False,
    )
