import asyncio
import json
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from livekit import rtc
from livekit.agents import llm
from livekit.agents.voice.events import ConversationItemAddedEvent, UserInputTranscribedEvent
from livekit.agents.voice.io import PlaybackFinishedEvent, PlaybackStartedEvent
from sqlalchemy import DefaultClause, Integer, MetaData, create_engine, func

from backend.core import deps
from backend.core.chat_history.postgres import PostgresChatHistoryStore, chat_messages
from backend.voice import agent


@pytest.fixture
def store(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{(tmp_path / 'voice-history.db').as_posix()}")
    table = chat_messages.to_metadata(MetaData())
    table.c.id.type = Integer()
    table.c.created_at.server_default = DefaultClause(func.current_timestamp())
    table.metadata.create_all(engine)
    monkeypatch.setattr(PostgresChatHistoryStore, "_engine", engine)
    history = PostgresChatHistoryStore()
    monkeypatch.setattr(deps, "_chat_history_store", history)
    try:
        yield history
    finally:
        engine.dispose()


def voice_session():
    session = rtc.EventEmitter()
    session.output = SimpleNamespace(audio=rtc.EventEmitter())
    session.current_speech = SimpleNamespace(id="speech-1")
    return session


def commit(session, role, content, **kwargs):
    message = llm.ChatMessage(role=role, content=[content], **kwargs)
    session.emit("conversation_item_added", ConversationItemAddedEvent(item=message))
    return message


def playback(session, text, *, interrupted=True, position=0.5):
    session.output.audio.emit("playback_started", PlaybackStartedEvent(created_at=1.0))
    session.output.audio.emit(
        "playback_finished",
        PlaybackFinishedEvent(
            playback_position=position,
            interrupted=interrupted,
            synchronized_transcript=text,
        ),
    )


@pytest.mark.parametrize("metadata", ["", " ", "{}", '{"playground": true}'])
def test_no_application_metadata(metadata):
    assert agent.application_session_id(metadata) is None


@pytest.mark.parametrize(
    "metadata",
    [
        "not-json",
        "[]",
        "null",
        '"text"',
        '{"session_id": null}',
        '{"session_id": 123}',
        '{"session_id": "invalid"}',
        '{"session_id": ""}',
    ],
)
def test_malformed_application_metadata(metadata):
    with pytest.raises(ValueError, match="Invalid application voice session metadata"):
        agent.application_session_id(metadata)


def test_loads_existing_text_history_and_does_not_reinsert_it(store):
    session_id = str(uuid4())
    store.add_message(session_id, "user", "Explain transformers")
    store.add_message(session_id, "assistant", "They use attention.")
    store.add_message("another-session", "user", "Private question")
    store.add_message(session_id, "system", "Do not import stored instructions")

    async def scenario():
        loaded_store, context = await agent.load_voice_history(session_id)
        assert loaded_store is store
        assert [(item.role, item.text_content) for item in context.items] == [
            ("user", "Explain transformers"),
            ("assistant", "They use attention."),
        ]
        session = voice_session()
        finish = agent.persist_voice_history(session, SimpleNamespace(shutdown=Mock()), store, session_id, context)
        for item in context.items:
            session.emit("conversation_item_added", ConversationItemAddedEvent(item=item))
        await finish()

    asyncio.run(scenario())
    assert len(store.get_messages(session_id)) == 3


def test_committed_messages_only_and_duplicate_ids(store):
    session_id = str(uuid4())

    async def scenario():
        session = voice_session()
        finish = agent.persist_voice_history(
            session, SimpleNamespace(shutdown=Mock()), store, session_id, llm.ChatContext()
        )
        for is_final in (False, True):
            session.emit(
                "user_input_transcribed", UserInputTranscribedEvent(transcript="STT fragment", is_final=is_final)
            )
        commit(session, "system", "Never persist instructions")
        commit(session, "developer", "Never persist developer instructions")
        commit(session, "assistant", " ")
        session.emit(
            "conversation_item_added",
            ConversationItemAddedEvent(item=llm.ChatMessage(role="user", content=[llm.AudioContent(frame=[])])),
        )
        message = commit(session, "user", "A committed question")
        session.emit("conversation_item_added", ConversationItemAddedEvent(item=message))
        commit(session, "assistant", "<think>Private reasoning</think>A spoken answer.", extra={"reasoning": "Secret"})
        # Repeating content with a new event ID is a legitimate new message.
        commit(session, "user", "A committed question")
        await finish()
        await finish()  # Shutdown is idempotent.

    asyncio.run(scenario())
    assert store.get_messages(session_id) == [
        {"role": "user", "content": "A committed question"},
        {"role": "assistant", "content": "A spoken answer."},
        {"role": "user", "content": "A committed question"},
    ]


def test_interrupted_response_uses_only_synchronized_spoken_segments(store):
    session_id = str(uuid4())

    async def scenario():
        session = voice_session()
        finish = agent.persist_voice_history(
            session, SimpleNamespace(shutdown=Mock()), store, session_id, llm.ChatContext()
        )
        playback(session, "Automatic greeting.", interrupted=False)
        # Starting a different speech prevents greeting playback from leaking in.
        session.current_speech = SimpleNamespace(id="reply-2")
        playback(session, "First sentence. ", interrupted=False)
        playback(session, "Partial second", interrupted=True)
        commit(
            session,
            "assistant",
            "First sentence. Full second sentence the user never heard.",
            interrupted=True,
            metrics={"started_speaking_at": 1.0},
        )
        await finish()

    asyncio.run(scenario())
    assert store.get_messages(session_id) == [{"role": "assistant", "content": "First sentence. Partial second"}]


@pytest.mark.parametrize(("synchronized", "position"), [(None, 0.5), ("", 0.5), ("Unheard", 0.0)])
def test_interruption_without_spoken_text_never_saves_full_generation(store, synchronized, position):
    session_id = str(uuid4())

    async def scenario():
        session = voice_session()
        finish = agent.persist_voice_history(
            session, SimpleNamespace(shutdown=Mock()), store, session_id, llm.ChatContext()
        )
        playback(session, synchronized, position=position)
        commit(session, "assistant", "Unheard full response", interrupted=True, metrics={"started_speaking_at": 1.0})
        await finish()

    asyncio.run(scenario())
    assert store.get_messages(session_id) == []


def test_database_writes_do_not_block_event_loop_and_shutdown_waits(store, monkeypatch):
    session_id = str(uuid4())
    started, release = threading.Event(), threading.Event()
    loop_thread = threading.get_ident()
    add_message = store.add_message

    def slow_write(*args):
        assert threading.get_ident() != loop_thread
        started.set()
        assert release.wait(timeout=5)
        add_message(*args)

    monkeypatch.setattr(store, "add_message", slow_write)

    async def scenario():
        session = voice_session()
        finish = agent.persist_voice_history(
            session, SimpleNamespace(shutdown=Mock()), store, session_id, llm.ChatContext()
        )
        commit(session, "user", "First")
        commit(session, "assistant", "Second")
        assert await asyncio.to_thread(started.wait, 5)
        shutdown = asyncio.create_task(finish())
        try:
            await asyncio.sleep(0.01)
            assert not shutdown.done()
        finally:
            release.set()
        await shutdown

    asyncio.run(scenario())
    assert [message["content"] for message in store.get_messages(session_id)] == ["First", "Second"]


def test_concurrent_sessions_use_explicit_session_ids(store):
    sessions = [str(uuid4()), str(uuid4())]

    async def scenario():
        finishes = []
        for session_id in sessions:
            session = voice_session()
            finishes.append(
                agent.persist_voice_history(
                    session, SimpleNamespace(shutdown=Mock()), store, session_id, llm.ChatContext()
                )
            )
            commit(session, "user", f"Question {session_id}")
            commit(session, "assistant", f"Answer {session_id}")
        await asyncio.gather(*(finish() for finish in finishes))

    asyncio.run(scenario())
    for session_id in sessions:
        assert [item["content"] for item in store.get_messages(session_id)] == [
            f"Question {session_id}",
            f"Answer {session_id}",
        ]


def test_load_failure_does_not_expose_database_details(monkeypatch):
    monkeypatch.setattr(deps, "get_chat_history_store", Mock(side_effect=RuntimeError("private DB credentials")))
    with pytest.raises(RuntimeError, match="Application voice history could not be loaded") as error:
        asyncio.run(agent.load_voice_history(str(uuid4())))
    assert "private DB credentials" not in str(error.value)


def test_write_failure_stops_session_without_unsafe_retries(store, monkeypatch, caplog):
    session_id = str(uuid4())
    failing = Mock(side_effect=RuntimeError("private SQL parameters and credentials"))
    monkeypatch.setattr(store, "add_message", failing)
    ctx = SimpleNamespace(shutdown=Mock())

    async def scenario():
        session = voice_session()
        finish = agent.persist_voice_history(session, ctx, store, session_id, llm.ChatContext())
        commit(session, "user", "Question")
        commit(session, "assistant", "Answer")
        with pytest.raises(RuntimeError, match="Application voice history could not be saved"):
            await finish()

    asyncio.run(scenario())
    assert failing.call_count == 1
    ctx.shutdown.assert_called_once_with(reason="voice history persistence failed")
    assert "private SQL parameters" not in caplog.text
    assert "2 voice messages were not confirmed saved" in caplog.text


@pytest.mark.parametrize("metadata", ["", '{"playground": true}', "application"])
def test_worker_startup_context_greeting_and_final_shutdown_messages(store, monkeypatch, metadata):
    session_id = str(uuid4())
    store.add_message(session_id, "user", "Existing text question")
    metadata = json.dumps({"session_id": session_id}) if metadata == "application" else metadata
    session = voice_session()
    session.start = AsyncMock()
    session.say = AsyncMock()
    session.aclose = AsyncMock(side_effect=lambda: commit(session, "user", "Final committed transcript"))
    components = [SimpleNamespace(aclose=AsyncMock()) for _ in range(3)]
    ctx = SimpleNamespace(
        job=SimpleNamespace(metadata=metadata),
        proc=SimpleNamespace(userdata={"vad": object()}),
        room=rtc.EventEmitter(),
        is_fake_job=lambda: True,
        connect=AsyncMock(),
        add_shutdown_callback=Mock(),
        shutdown=Mock(),
    )
    monkeypatch.setattr(agent, "configure", lambda: None)
    monkeypatch.setattr(agent, "setup_logging", lambda: None)
    monkeypatch.setattr(agent, "diagnostics", lambda *_: None)
    monkeypatch.setattr(agent, "providers", lambda: components)
    monkeypatch.setattr(agent, "MultilingualModel", lambda: object())
    monkeypatch.setattr(agent, "AgentSession", lambda **_: session)
    if not metadata or "playground" in metadata:
        monkeypatch.setattr(
            deps, "get_chat_history_store", Mock(side_effect=AssertionError("Standalone must not access DB"))
        )

    async def scenario():
        await agent.entrypoint(ctx)
        loaded = session.start.call_args.kwargs["agent"].chat_ctx.items
        assert [item.text_content for item in loaded] == (
            ["Existing text question"] if "session_id" in metadata else []
        )
        assert session.start.call_args.kwargs["record"] is False
        assert session.say.call_args.kwargs["add_to_chat_ctx"] is False
        await ctx.add_shutdown_callback.call_args.args[0]()
        for component in components:
            component.aclose.assert_awaited_once()

    asyncio.run(scenario())
    expected = (
        ["Existing text question", "Final committed transcript"]
        if "session_id" in metadata
        else ["Existing text question"]
    )
    assert [item["content"] for item in store.get_messages(session_id)] == expected


def test_invalid_metadata_closes_job_before_providers_start(monkeypatch):
    ctx = SimpleNamespace(job=SimpleNamespace(metadata='{"session_id": "bad"}'), shutdown=Mock())
    providers = Mock()
    monkeypatch.setattr(agent, "setup_logging", lambda: None)
    monkeypatch.setattr(agent, "configure", lambda: None)
    monkeypatch.setattr(agent, "providers", providers)
    with pytest.raises(ValueError, match="Invalid application voice session metadata"):
        asyncio.run(agent.entrypoint(ctx))
    providers.assert_not_called()
    ctx.shutdown.assert_called_once()


def test_database_load_failure_closes_job_before_providers_start(monkeypatch):
    ctx = SimpleNamespace(job=SimpleNamespace(metadata=json.dumps({"session_id": str(uuid4())})), shutdown=Mock())
    providers = Mock()
    monkeypatch.setattr(agent, "setup_logging", lambda: None)
    monkeypatch.setattr(agent, "configure", lambda: None)
    monkeypatch.setattr(agent, "providers", providers)
    monkeypatch.setattr(deps, "get_chat_history_store", Mock(side_effect=RuntimeError("private DB connection")))
    with pytest.raises(RuntimeError, match="Application voice history could not be loaded"):
        asyncio.run(agent.entrypoint(ctx))
    providers.assert_not_called()
    ctx.shutdown.assert_called_once()
