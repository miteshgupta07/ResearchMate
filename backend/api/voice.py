"""Room-scoped voice tokens, explicit worker dispatch, and local abuse protection."""

import json
import math
import os
from datetime import timedelta
from pathlib import Path
from threading import Lock
from time import monotonic
from urllib.parse import urlsplit
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException, Request
from livekit import api
from starlette.responses import JSONResponse, Response

from ..core.deps import get_exclusive_history_store

from ..schemas.voice import VoiceSessionRequest, VoiceSessionResponse

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env", override=False)

router = APIRouter(prefix="/voice", tags=["Voice"])
TOKEN_TTL = timedelta(minutes=5)
DEFAULT_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173"
_rate_lock = Lock()
_rate_started = 0.0
_rate_total = 0
_rate_clients: dict[str, int] = {}


async def voice_origin_guard(request: Request, call_next):
    """Override the application's permissive CORS policy only for this endpoint."""
    path = request.scope["path"]
    root_path = request.scope.get("root_path", "").rstrip("/")
    if root_path and path.startswith(root_path + "/"):
        path = path[len(root_path) :]
    if path.rstrip("/") != "/voice/session":
        return await call_next(request)

    allowed = {origin.strip() for origin in os.getenv("VOICE_ALLOWED_ORIGINS", DEFAULT_ORIGINS).split(",")}
    origin = request.headers.get("origin")
    if not origin or origin == "null" or origin not in allowed:
        return JSONResponse(
            status_code=403,
            content={"detail": "Voice origin is not allowed."},
            headers={"Cache-Control": "no-store"},
        )

    if request.method == "OPTIONS":
        method = request.headers.get("access-control-request-method", "")
        headers = {
            value.strip().lower()
            for value in request.headers.get("access-control-request-headers", "").split(",")
            if value.strip()
        }
        if method != "POST" or not headers <= {"content-type"}:
            response = JSONResponse(status_code=403, content={"detail": "Voice preflight is not allowed."})
        else:
            response = Response(
                status_code=204,
                headers={
                    "Access-Control-Allow-Methods": "POST",
                    "Access-Control-Allow-Headers": "Content-Type",
                },
            )
    else:
        response = await call_next(request)

    response.headers["Access-Control-Allow-Origin"] = origin
    response.headers["Cache-Control"] = "no-store"
    response.headers["Pragma"] = "no-cache"
    response.headers.add_vary_header("Origin")
    return response


def check_rate_limit(request: Request) -> None:
    """Bound token creation per client and globally, per server process/window.

    Use the server's client address, never a caller-supplied forwarding header.
    The global cap also bounds the number of tracked client addresses.
    """
    try:
        per_client = int(os.getenv("VOICE_RATE_LIMIT_REQUESTS", "5"))
        total = int(os.getenv("VOICE_RATE_LIMIT_GLOBAL_REQUESTS", "60"))
        window = int(os.getenv("VOICE_RATE_LIMIT_WINDOW_SECONDS", "60"))
        if not (1 <= per_client <= total <= 10000 and 1 <= window <= 3600):
            raise ValueError
    except ValueError:
        raise HTTPException(status_code=503, detail="Voice service configuration is unavailable.") from None

    global _rate_started, _rate_total
    client = request.client.host if request.client else "unknown"
    with _rate_lock:
        now = monotonic()
        if now - _rate_started >= window:
            _rate_started = now
            _rate_total = 0
            _rate_clients.clear()
        if _rate_total >= total or _rate_clients.get(client, 0) >= per_client:
            raise HTTPException(
                status_code=429,
                detail="Too many voice session requests. Try again later.",
                headers={"Retry-After": str(max(1, math.ceil(window - (now - _rate_started))))},
            )
        _rate_total += 1
        _rate_clients[client] = _rate_clients.get(client, 0) + 1


@router.post("/session", response_model=VoiceSessionResponse, dependencies=[Depends(get_exclusive_history_store)])
def create_voice_session(body: VoiceSessionRequest, request: Request) -> VoiceSessionResponse:
    url = os.getenv("LIVEKIT_URL", "").strip()
    key = os.getenv("LIVEKIT_API_KEY", "").strip()
    secret = os.getenv("LIVEKIT_API_SECRET", "").strip()
    try:
        parsed = urlsplit(url)
        if not key or not secret or parsed.scheme != "wss" or not parsed.hostname:
            raise ValueError
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError
        parsed.port  # Reject malformed ports without exposing the configured URL.
    except ValueError:
        raise HTTPException(status_code=503, detail="Voice service configuration is unavailable.") from None

    check_rate_limit(request)
    room_name = f"researchmate-voice-{uuid4().hex}"
    identity = f"voice-user-{uuid4().hex}"
    try:
        token = (
            api.AccessToken(key, secret)
            .with_identity(identity)
            .with_ttl(TOKEN_TTL)
            .with_grants(
                api.VideoGrants(
                    room_join=True,
                    room=room_name,
                    can_publish=True,
                    can_subscribe=True,
                    can_publish_sources=["microphone"],
                    can_publish_data=False,
                    can_update_own_metadata=False,
                    room_create=False,
                    room_list=False,
                    room_record=False,
                    room_admin=False,
                    ingress_admin=False,
                    agent=False,
                    can_manage_agent_session=False,
                    can_subscribe_metrics=False,
                )
            )
            .with_room_config(
                api.RoomConfiguration(
                    name=room_name,
                    agents=[
                        api.RoomAgentDispatch(
                            agent_name="researchmate-voice",
                            metadata=json.dumps({"session_id": str(body.session_id)}),
                        )
                    ],
                )
            )
            .to_jwt()
        )
    except Exception:
        # Do not log provider exceptions, signing inputs, credentials, or tokens.
        raise HTTPException(status_code=503, detail="Voice session could not be created.") from None
    return VoiceSessionResponse(url=url, room_name=room_name, token=token)
