from contextlib import nullcontext

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from sqlalchemy import DefaultClause, Integer, MetaData, create_engine, func

from backend.api import agent, chat, portfolio, rag
from backend.core import deps
from backend.core import rag as rag_core
from backend.core.chat_history.postgres import PostgresChatHistoryStore, chat_messages
from backend.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


@pytest.fixture
def history_store(tmp_path, monkeypatch):
    # Exercise the production SQLAlchemy queries against an isolated test DB.
    # Only test DDL adapts PostgreSQL's BIGSERIAL/NOW() to SQLite equivalents.
    engine = create_engine(f"sqlite:///{(tmp_path / 'history.db').as_posix()}")
    table = chat_messages.to_metadata(MetaData())
    table.c.id.type = Integer()
    table.c.created_at.server_default = DefaultClause(func.current_timestamp())
    table.metadata.create_all(engine)
    monkeypatch.setattr(PostgresChatHistoryStore, "_engine", engine)
    store = PostgresChatHistoryStore()
    monkeypatch.setattr(deps, "_chat_history_store", store)
    monkeypatch.setattr(store, "session_lock", lambda *_args, **_kwargs: nullcontext(store))
    try:
        yield store
    finally:
        engine.dispose()


def test_interleaved_history_operations(history_store):
    store = history_store
    assert deps.get_chat_history_store() is store
    assert PostgresChatHistoryStore()._engine is store._engine
    store.add_message("alpha", "user", "alpha question")
    store.add_message("beta", "user", "beta question")
    store.add_message("alpha", "assistant", "alpha answer")
    store.add_message("beta", "system", "beta instructions")

    assert store.get_langchain_messages("alpha") == [("human", "alpha question"), ("ai", "alpha answer")]
    assert store.get_langchain_messages("beta") == [("human", "beta question"), ("system", "beta instructions")]
    assert store.get_messages("alpha") == store.get_session_messages("alpha")
    assert store.get_messages("unknown") == []

    store.clear("alpha")
    assert store.get_messages("alpha") == []
    assert len(store.get_messages("beta")) == 2
    store.clear_session("beta")
    assert store.get_session_messages("beta") == []


def test_concurrent_store_operations(history_store):
    barrier = Barrier(4)

    def turn(session_id):
        history_store.add_message(session_id, "user", session_id)
        barrier.wait(timeout=10)
        assert history_store.get_langchain_messages(session_id) == [("human", session_id)]
        history_store.add_message(session_id, "assistant", f"reply-{session_id}")
        barrier.wait(timeout=10)
        if session_id == "alpha":
            history_store.clear(session_id)
        barrier.wait(timeout=10)
        expected = [] if session_id == "alpha" else [
            {"role": "user", "content": session_id},
            {"role": "assistant", "content": f"reply-{session_id}"},
        ]
        assert history_store.get_messages(session_id) == expected

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(turn, ("alpha", "beta", "gamma", "delta")))


@pytest.mark.parametrize(
    ("path", "intent"),
    [
        ("/chat", None),
        ("/rag/query", None),
        ("/agent/route", agent.INTENT_CHAT),
        ("/agent/route", agent.INTENT_RAG_QA),
        ("/agent/route", agent.INTENT_RAG_SUMMARY),
        ("/portfolio/chat", "greeting"),
        ("/portfolio/chat", "followup"),
        ("/portfolio/chat", "profile_query"),
    ],
)
def test_concurrent_api_history_isolation(path, intent, history_store, monkeypatch):
    barrier = Barrier(2)
    summary_prompt = (
        "Please provide a comprehensive summary of this document, highlighting the key points, "
        "main arguments, and conclusions."
    )

    def user_message(session_id):
        return summary_prompt if intent == agent.INTENT_RAG_SUMMARY else f"question-{session_id}"

    def answer(**kwargs):
        history = kwargs["chat_history_langchain"]
        session_id = history[0][1].removeprefix("prior-")
        assert history == [("ai", f"prior-{session_id}"), ("human", user_message(session_id))]
        assert kwargs["language"] == "Hindi"
        if "user_input" in kwargs:
            assert kwargs["user_input"] == user_message(session_id)
            assert kwargs["retriever"] is retriever
        barrier.wait(timeout=10)  # Both requests pause before persisting their replies.
        return f"reply-{session_id}"

    def portfolio_answer(prompt):
        messages = prompt.to_messages()
        session_id = messages[1].content.removeprefix("prior-")
        assert len(messages) == 3
        assert messages[1].type == "ai"
        if intent == "profile_query":
            assert messages[-1].content == f"Context:\ndocument-context\n\nQuestion: question-{session_id}"
        else:
            assert messages[-1].content == f"question-{session_id}"
        barrier.wait(timeout=10)
        return AIMessage(content=f"reply-{session_id}")

    model = RunnableLambda(portfolio_answer)

    def get_model(**kwargs):
        if path != "/portfolio/chat":
            assert kwargs == {"model_type": "GPT OSS", "temperature": 0.4, "max_tokens": 99}
        return model

    retriever = SimpleNamespace(invoke=lambda _: [SimpleNamespace(page_content="document-context")])
    monkeypatch.setattr(app.state, "portfolio_retriever", retriever, raising=False)
    for module in (chat, rag, agent, portfolio):
        monkeypatch.setattr(module, "get_llm", get_model)
    monkeypatch.setattr(rag_core, "answer_without_rag", answer)
    monkeypatch.setattr(rag_core, "answer_query", answer)
    monkeypatch.setattr(rag_core, "load_retriever", lambda _: retriever)
    monkeypatch.setattr(agent, "prechecks", lambda *_: (intent, 0.9))
    monkeypatch.setattr(portfolio, "classify_intent", lambda _: intent)

    for session_id in ("alpha", "beta"):
        history_store.add_message(session_id, "assistant", f"prior-{session_id}")

    def post(session_id):
        payload = {"session_id": session_id, "message": f"question-{session_id}"}
        if path != "/portfolio/chat":
            payload.update(language="Hindi", model_type="GPT OSS", temperature=0.4, max_tokens=99)
        if path in ("/rag/query", "/agent/route"):
            payload["document_id"] = "document-123"
        response = client.post(path, json=payload)
        assert response.status_code == 200, response.text
        expected = {"role": "assistant", "content": f"reply-{session_id}"}
        if path == "/rag/query":
            expected["sources"] = []
        elif path == "/agent/route":
            metadata = {} if intent == agent.INTENT_CHAT else {"document_id": "document-123"}
            if intent == agent.INTENT_RAG_SUMMARY:
                metadata["type"] = "summary"
            expected.update(intent=intent, confidence=0.9, metadata=metadata)
        elif path == "/portfolio/chat":
            expected = {"response": f"reply-{session_id}"}
        assert response.json() == expected

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(post, ("alpha", "beta")))

    expected_histories = {
        session_id: [
            {"role": "assistant", "content": f"prior-{session_id}"},
            {"role": "user", "content": user_message(session_id)},
            {"role": "assistant", "content": f"reply-{session_id}"},
        ]
        for session_id in ("alpha", "beta")
    }
    for session_id in expected_histories:
        response = client.get("/history", params={"session_id": session_id})
        assert response.status_code == 200
        assert response.json() == {"session_id": session_id, "messages": expected_histories[session_id]}

    response = client.delete("/history", params={"session_id": "alpha"})
    assert response.status_code == 200
    assert response.json() == {"status": "cleared"}
    assert client.get("/history", params={"session_id": "alpha"}).json() == {"session_id": "alpha", "messages": []}
    assert client.get("/history", params={"session_id": "beta"}).json()["messages"] == expected_histories["beta"]


@pytest.mark.parametrize("method", ["get", "delete"])
@pytest.mark.parametrize("params", [{}, {"session_id": ""}])
def test_history_api_requires_session_id(method, params, history_store):
    assert getattr(client, method)("/history", params=params).status_code == 422
