"""Opt-in tests on the configured PostgreSQL, using only disposable session IDs."""

import asyncio
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from livekit import rtc
from livekit.agents import llm
from livekit.agents.voice.events import ConversationItemAddedEvent

from backend.api import chat
from backend.core import deps
from backend.core import rag as rag_core
from backend.core.chat_history.postgres import PostgresChatHistoryStore
from backend.main import app
from backend.voice import agent

pytestmark = pytest.mark.skipif(os.getenv("RUN_POSTGRES_TESTS") != "1", reason="Requires live PostgreSQL")


@pytest.fixture
def database(monkeypatch):
    store = PostgresChatHistoryStore()
    monkeypatch.setattr(deps, "_chat_history_store", store)
    session_ids = []

    def new_session():
        session_id = str(uuid4())
        session_ids.append(session_id)
        return session_id

    yield store, new_session
    for session_id in session_ids:
        with store.session_lock(session_id) as locked:
            locked.clear_session(session_id)


def assert_busy(store, session_id):
    with pytest.raises(HTTPException) as failure:
        with store.session_lock(session_id):
            pytest.fail("A competing connection acquired the lock")
    assert failure.value.status_code == 409


def test_text_voice_clear_and_completion_barrier(database, monkeypatch):
    store, new_session = database
    session_id, independent = new_session(), new_session()
    started, release = Event(), Event()
    monkeypatch.setattr(chat, "get_llm", lambda **_: object())

    def answer(**kwargs):
        if kwargs["chat_history_langchain"][-1][1] == "hold":
            started.set()
            assert release.wait(10)
        return "complete answer"

    monkeypatch.setattr(rag_core, "answer_without_rag", answer)
    client = TestClient(app)
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            running = executor.submit(client.post, "/chat", json={"session_id": session_id, "message": "hold"})
            try:
                assert started.wait(10)
                for path in ("/chat", "/rag/query", "/agent/route", "/portfolio/chat"):
                    response = client.post(path, json={"session_id": session_id, "message": "conflict", "document_id": "test"})
                    assert response.status_code == 409, response.text
                assert client.post("/voice/session", json={"session_id": session_id}, headers={"Origin": "http://localhost:5173"}).status_code == 409
                assert client.delete("/history", params={"session_id": session_id}).status_code == 409
                assert client.get("/history", params={"session_id": session_id}).status_code == 200
                assert client.post("/chat", json={"session_id": independent, "message": "independent"}).status_code == 200
                waiting = executor.submit(client.get, "/history", params={"session_id": session_id, "wait_for_idle": True})
                assert not waiting.done()
            finally:
                release.set()
            assert running.result(10).status_code == 200
            snapshot = waiting.result(10)
            assert snapshot.status_code == 200
            assert snapshot.json()["messages"] == [
                {"role": "user", "content": "hold"}, {"role": "assistant", "content": "complete answer"},
            ]
        assert store.get_messages(independent)[-1]["content"] == "complete answer"
        assert client.delete("/history", params={"session_id": session_id}).status_code == 200
    finally:
        client.close()


def test_voice_lock_rejects_duplicate_worker_and_text(database, monkeypatch):
    store, new_session = database
    session_id, independent = new_session(), new_session()
    providers = Mock()
    monkeypatch.setattr(agent, "configure", lambda: None)
    monkeypatch.setattr(agent, "setup_logging", lambda: None)
    monkeypatch.setattr(agent, "providers", providers)
    ctx = SimpleNamespace(job=SimpleNamespace(metadata=json.dumps({"session_id": session_id})), shutdown=Mock(), add_shutdown_callback=Mock())
    client = TestClient(app)
    try:
        with store.session_lock(session_id) as voice_store:
            voice_store.add_message(session_id, "user", "voice turn")
            assert client.post("/chat", json={"session_id": session_id, "message": "blocked"}).status_code == 409
            assert client.post("/voice/session", json={"session_id": session_id}, headers={"Origin": "http://localhost:5173"}).status_code == 409
            with pytest.raises(HTTPException) as failure:
                asyncio.run(agent.entrypoint(ctx))
            assert failure.value.status_code == 409
            providers.assert_not_called()
            with store.session_lock(independent):
                pass
        assert store.get_messages(session_id) == [{"role": "user", "content": "voice turn"}]
        with store.session_lock(session_id):
            pass
    finally:
        client.close()


def test_process_termination_releases_postgres_lock(database):
    store, new_session = database
    session_id = new_session()
    script = (
        "from backend.core.chat_history.postgres import PostgresChatHistoryStore\n"
        "store = PostgresChatHistoryStore()\n"
        f"with store.session_lock({session_id!r}):\n"
        " print('locked', flush=True)\n"
        " input()\n"
    )
    process = subprocess.Popen([sys.executable, "-c", script], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    with ThreadPoolExecutor(max_workers=1) as executor:
        try:
            assert executor.submit(process.stdout.readline).result(30).strip() == "locked"
            assert_busy(store, session_id)
        finally:
            process.terminate()
            process.wait(10)
            process.stdin.close()
            process.stdout.close()
            process.stderr.close()
    assert process.poll() is not None
    # Process exit precedes the server observing the closed socket. Wait on PG,
    # rather than sleeping or assuming that notification is instantaneous.
    with store.session_lock(session_id, wait=True):
        pass


def test_failure_and_lost_connection_do_not_leak_or_reconnect_without_lock(database):
    store, new_session = database
    session_id = new_session()
    with pytest.raises(RuntimeError, match="generation failed"):
        with store.session_lock(session_id):
            raise RuntimeError("generation failed")
    with store.session_lock(session_id) as locked:
        locked._connection.invalidate()
        with pytest.raises(HTTPException):
            locked.add_message(session_id, "user", "must not be saved")
    with store.session_lock(session_id):
        assert store.get_messages(session_id) == []


def test_shutdown_holds_lock_until_final_voice_write_finishes(database, monkeypatch):
    store, new_session = database
    session_id = new_session()
    started, release = Event(), Event()
    insert = PostgresChatHistoryStore.add_message

    def slow_insert(self, *args):
        started.set()
        assert release.wait(10)
        insert(self, *args)

    monkeypatch.setattr(PostgresChatHistoryStore, "add_message", slow_insert)
    session = rtc.EventEmitter()
    session.output = SimpleNamespace(audio=rtc.EventEmitter())
    session.current_speech = None
    session.start, session.say = AsyncMock(), AsyncMock()
    message = llm.ChatMessage(role="user", content=["final voice transcript"])
    session.aclose = AsyncMock(side_effect=lambda: session.emit("conversation_item_added", ConversationItemAddedEvent(item=message)))
    components = [SimpleNamespace(aclose=AsyncMock()) for _ in range(3)]
    ctx = SimpleNamespace(job=SimpleNamespace(metadata=json.dumps({"session_id": session_id})), proc=SimpleNamespace(userdata={"vad": object()}), room=rtc.EventEmitter(), is_fake_job=lambda: True, add_shutdown_callback=Mock(), shutdown=Mock())
    monkeypatch.setattr(agent, "configure", lambda: None)
    monkeypatch.setattr(agent, "setup_logging", lambda: None)
    monkeypatch.setattr(agent, "diagnostics", lambda *_: None)
    monkeypatch.setattr(agent, "MultilingualModel", lambda: object())
    monkeypatch.setattr(agent, "providers", lambda: components)
    monkeypatch.setattr(agent, "AgentSession", lambda **_: session)

    async def scenario():
        await agent.entrypoint(ctx)
        finish = ctx.add_shutdown_callback.call_args.args[0]
        shutdown = asyncio.create_task(finish())
        try:
            assert await asyncio.to_thread(started.wait, 5)
            await asyncio.to_thread(assert_busy, store, session_id)
            shutdown.cancel()
            with pytest.raises(asyncio.CancelledError):
                await shutdown
            await asyncio.to_thread(assert_busy, store, session_id)
        finally:
            release.set()
        await finish()

    asyncio.run(scenario())
    with store.session_lock(session_id):
        assert store.get_messages(session_id) == [{"role": "user", "content": "final voice transcript"}]


def test_cancelled_worker_startup_finishes_lock_checkout_before_cleanup(database, monkeypatch):
    store, new_session = database
    session_id = new_session()
    started, release = Event(), Event()
    acquire = store.session_lock

    @contextmanager
    def delayed_acquire(*args, **kwargs):
        started.set()
        assert release.wait(10)
        with acquire(*args, **kwargs) as locked:
            yield locked

    monkeypatch.setattr(store, "session_lock", delayed_acquire)
    monkeypatch.setattr(agent, "configure", lambda: None)
    monkeypatch.setattr(agent, "setup_logging", lambda: None)
    ctx = SimpleNamespace(job=SimpleNamespace(metadata=json.dumps({"session_id": session_id})), add_shutdown_callback=Mock(), shutdown=Mock())

    async def scenario():
        startup = asyncio.create_task(agent.entrypoint(ctx))
        try:
            assert await asyncio.to_thread(started.wait, 5)
            startup.cancel()
            with pytest.raises(asyncio.CancelledError):
                await startup
            shutdown = asyncio.create_task(ctx.add_shutdown_callback.call_args.args[0]())
        finally:
            release.set()
        await shutdown

    asyncio.run(scenario())
    with acquire(session_id):
        pass
