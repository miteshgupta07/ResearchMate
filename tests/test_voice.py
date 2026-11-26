from contextlib import nullcontext
from types import SimpleNamespace
from backend.core import deps
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from unittest.mock import Mock
from uuid import uuid4

import jwt
import pytest
from fastapi.testclient import TestClient
from livekit import api

from backend.api import voice
from backend.main import app
from backend.schemas.voice import VoiceSessionResponse

ORIGIN = "https://researchmate.test"
KEY = "test-livekit-key"
SECRET = "test-livekit-secret-with-at-least-32-characters"


@pytest.fixture
def client(monkeypatch):
    for name, value in {
        "LIVEKIT_URL": "wss://voice.example.test",
        "LIVEKIT_API_KEY": KEY,
        "LIVEKIT_API_SECRET": SECRET,
        "VOICE_ALLOWED_ORIGINS": ORIGIN,
        "VOICE_RATE_LIMIT_REQUESTS": "5",
        "VOICE_RATE_LIMIT_GLOBAL_REQUESTS": "60",
        "VOICE_RATE_LIMIT_WINDOW_SECONDS": "60",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(voice, "_rate_started", 1000.0)
    monkeypatch.setattr(voice, "_rate_total", 0)
    monkeypatch.setattr(voice, "_rate_clients", {})
    monkeypatch.setattr(voice, "monotonic", lambda: 1000.0)
    app.dependency_overrides[deps.get_chat_history_store] = lambda: SimpleNamespace(session_lock=lambda *_: nullcontext(None))
    connection = TestClient(app, headers={"Origin": ORIGIN})
    try:
        yield connection
    finally:
        connection.close()
        app.dependency_overrides.clear()


def create_session(client, session_id=None, **kwargs):
    return client.post("/voice/session", json={"session_id": session_id or str(uuid4())}, **kwargs)


def decode_token(response):
    return jwt.decode(
        response.json()["token"],
        SECRET,
        algorithms=["HS256"],
        issuer=KEY,
        options={"require": ["exp", "nbf", "iss", "sub"]},
    )


def test_voice_token_permissions_dispatch_and_metadata(client, caplog):
    caplog.set_level(logging.DEBUG)
    session_id = str(uuid4())
    response = create_session(client, session_id)
    assert response.status_code == 200
    data = response.json()
    assert set(data) == {"url", "room_name", "token"}
    assert data["url"] == "wss://voice.example.test"
    assert data["room_name"].startswith("researchmate-voice-")
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["access-control-allow-origin"] == ORIGIN
    claims = decode_token(response)
    assert claims["sub"].startswith("voice-user-")
    assert 299 <= claims["exp"] - claims["nbf"] <= 301
    assert claims["video"] == {
        "roomJoin": True,
        "room": data["room_name"],
        "canPublish": True,
        "canSubscribe": True,
        "canPublishSources": ["microphone"],
        "canPublishData": False,
        "canUpdateOwnMetadata": False,
        "roomCreate": False,
        "roomList": False,
        "roomRecord": False,
        "roomAdmin": False,
        "ingressAdmin": False,
        "agent": False,
        "canManageAgentSession": False,
        "canSubscribeMetrics": False,
    }
    assert (
        not {"sip", "inference", "observability", "agent", "agentEndpoint", "agentThread", "agentDatabase"}
        & claims.keys()
    )
    assert claims["roomConfig"]["name"] == data["room_name"]
    (dispatch,) = claims["roomConfig"]["agents"]
    assert dispatch["agentName"] == "researchmate-voice"
    assert json.loads(dispatch["metadata"]) == {"session_id": session_id}
    verified = api.TokenVerifier(KEY, SECRET).verify(data["token"])
    assert verified.room_config.agents[0].agent_name == "researchmate-voice"
    assert data["token"] not in repr(VoiceSessionResponse(**data))
    for sensitive in (data["token"], KEY, SECRET):
        assert sensitive not in caplog.text


def test_voice_token_expires(client, monkeypatch):
    response = create_session(client)
    claims = decode_token(response)
    clock = Mock(wraps=datetime)
    clock.now.return_value = datetime.fromtimestamp(claims["exp"] + 1, tz=timezone.utc)
    monkeypatch.setattr(jwt.api_jwt, "datetime", clock)
    with pytest.raises(jwt.ExpiredSignatureError):
        decode_token(response)


def test_connections_have_unique_rooms_and_identities(client):
    session_id = str(uuid4())
    first, second = [create_session(client, session_id) for _ in range(2)]
    assert first.status_code == second.status_code == 200
    assert first.json()["room_name"] != second.json()["room_name"]
    assert decode_token(first)["sub"] != decode_token(second)["sub"]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"session_id": ""},
        {"session_id": "not-a-uuid"},
        {"session_id": None},
        {"session_id": str(uuid4()), "room_name": "another-room"},
        {"session_id": str(uuid4()), "metadata": {"session_id": "injected"}},
    ],
)
def test_invalid_session_request(client, payload):
    assert client.post("/voice/session", json=payload).status_code == 422
    assert voice._rate_total == 0


@pytest.mark.parametrize("name", ["LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"])
def test_missing_configuration(client, monkeypatch, name, caplog):
    monkeypatch.setenv(name, "")
    response = create_session(client)
    assert response.status_code == 503
    assert response.json() == {"detail": "Voice service configuration is unavailable."}
    assert voice._rate_total == 0
    assert SECRET not in caplog.text


@pytest.mark.parametrize(
    "url",
    [
        "https://voice.test",
        "wss://user:password@voice.test",
        "wss://voice.test?token=secret",
        "wss://voice.test:invalid",
    ],
)
def test_invalid_server_url(client, monkeypatch, url):
    monkeypatch.setenv("LIVEKIT_URL", url)
    assert create_session(client).status_code == 503


def test_signing_failure_does_not_expose_secrets(client, monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    sensitive_error = f"{KEY} {SECRET} generated-token-must-not-be-logged"
    monkeypatch.setattr(voice.api, "AccessToken", Mock(side_effect=RuntimeError(sensitive_error)))
    response = create_session(client)
    assert response.status_code == 503
    assert response.json() == {"detail": "Voice session could not be created."}
    for sensitive in (KEY, SECRET, "generated-token-must-not-be-logged"):
        assert sensitive not in response.text
        assert sensitive not in caplog.text


@pytest.mark.parametrize("origin", ["https://untrusted.test", "https://researchmate.test.evil.test", "null", "*"])
def test_disallowed_origin(client, origin):
    response = create_session(client, headers={"Origin": origin})
    assert response.status_code == 403
    assert "access-control-allow-origin" not in response.headers
    assert voice._rate_total == 0


def test_missing_origin_rejected(client):
    client.headers.pop("Origin")
    assert create_session(client).status_code == 403


def test_origin_guard_with_deployment_prefix(client):
    connection = TestClient(app, root_path="/api", headers={"Origin": "https://untrusted.test"})
    try:
        response = connection.post("/api/voice/session", json={"session_id": str(uuid4())})
        assert response.status_code == 403
        assert voice._rate_total == 0
    finally:
        connection.close()
        app.dependency_overrides.clear()


def test_voice_preflight_overrides_global_cors(client):
    headers = {"Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type"}
    response = client.options("/voice/session", headers=headers)
    assert response.status_code == 204
    assert response.headers["access-control-allow-origin"] == ORIGIN
    assert response.headers["access-control-allow-methods"] == "POST"
    assert voice._rate_total == 0
    assert client.options("/voice/session", headers={**headers, "Origin": "https://evil.test"}).status_code == 403
    assert (
        client.options(
            "/voice/session", headers={**headers, "Access-Control-Request-Headers": "x-unapproved"}
        ).status_code
        == 403
    )
    assert (
        client.options("/voice/session", headers={**headers, "Access-Control-Request-Method": "DELETE"}).status_code
        == 403
    )
    # Existing routes retain their CORS behavior and need no voice-origin header.
    assert client.options("/chat", headers={**headers, "Origin": "https://evil.test"}).status_code == 200
    client.headers.pop("Origin")
    assert client.get("/health").status_code == 200


def test_default_local_origins(client, monkeypatch):
    monkeypatch.delenv("VOICE_ALLOWED_ORIGINS")
    for origin in voice.DEFAULT_ORIGINS.split(","):
        assert create_session(client, headers={"Origin": origin}).status_code == 200
    assert create_session(client).status_code == 403


def test_rate_limit_is_atomic_and_recovers(client, monkeypatch):
    monkeypatch.setenv("VOICE_RATE_LIMIT_REQUESTS", "2")
    with ThreadPoolExecutor(max_workers=8) as executor:
        responses = list(executor.map(lambda _: create_session(client), range(8)))
    assert [response.status_code for response in responses].count(200) == 2
    rejected = [response for response in responses if response.status_code == 429]
    assert len(rejected) == 6
    assert all(response.headers["retry-after"] == "60" for response in rejected)
    assert create_session(client, headers={"X-Forwarded-For": "198.51.100.10"}).status_code == 429
    monkeypatch.setattr(voice, "monotonic", lambda: 1060.0)
    assert create_session(client).status_code == 200


def test_global_rate_limit_bounds_clients(client, monkeypatch):
    monkeypatch.setenv("VOICE_RATE_LIMIT_REQUESTS", "1")
    monkeypatch.setenv("VOICE_RATE_LIMIT_GLOBAL_REQUESTS", "2")
    for index in range(3):
        connection = TestClient(app, client=(f"192.0.2.{index}", 50000), headers={"Origin": ORIGIN})
        try:
            assert create_session(connection).status_code == (200 if index < 2 else 429)
        finally:
            connection.close()
    assert len(voice._rate_clients) == 2


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("VOICE_RATE_LIMIT_REQUESTS", "0"),
        ("VOICE_RATE_LIMIT_REQUESTS", "invalid"),
        ("VOICE_RATE_LIMIT_GLOBAL_REQUESTS", "1"),
        ("VOICE_RATE_LIMIT_GLOBAL_REQUESTS", "10001"),
        ("VOICE_RATE_LIMIT_WINDOW_SECONDS", "0"),
    ],
)
def test_invalid_rate_configuration(client, monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    assert create_session(client).status_code == 503
