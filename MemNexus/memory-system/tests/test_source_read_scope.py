"""Whole-event history reads require an explicit request through both transports."""

import asyncio
import json

import pytest
from fastapi.testclient import TestClient
from test_sdk_bridge import bridge  # noqa: F401 - shared optional-SDK fixture

from agent_memory.api import AnswerQuery, create_app
from agent_memory.models import AnswerResponse, Scope
from agent_memory.runtime import MemoryRuntime
from agent_memory.settings import Settings


@pytest.mark.parametrize("read_scope", ["quotes", "events"])
def test_http_and_stream_forward_the_same_read_scope(tmp_path, monkeypatch, read_scope):
    runtime = MemoryRuntime(Settings(db_path=tmp_path / "scope.db"))
    calls = []

    async def answer(*args, **kwargs):
        calls.append(kwargs)
        return AnswerResponse(generated_text="recorded answer", citations=[], sources=[],
                              retrieval_count=0, elapsed_ms=0)

    monkeypatch.setattr(runtime, "answer", answer)
    try:
        with TestClient(create_app(runtime=runtime)) as client:
            sid = client.post("/api/v1/sessions", json={}).json()["session_id"]
            body = {"session_id": sid, "query": "Which recipe?", "source_session_ids": ["past"],
                    "source_read_scope": read_scope}
            assert client.post("/api/v1/answer", json=body).status_code == 200
            stream = client.post(f"/api/v1/sessions/{sid}/answer/stream", json=body)
            assert json.loads(stream.text.splitlines()[-1])["type"] == "result"
            assert len(calls) == 2
            assert all(call["source_read_scope"] == read_scope for call in calls)
            assert all(call["source_session_ids"] == ("past",) for call in calls)
    finally:
        runtime.retriever.close()
        runtime.store.close()


def test_http_default_is_quote_only_and_unknown_scopes_are_rejected():
    assert AnswerQuery(session_id="s", query="q").source_read_scope == "quotes"
    with pytest.raises(ValueError):
        AnswerQuery(session_id="s", query="q", source_read_scope="all")


def test_sdk_event_permission_is_explicit_and_context_local(bridge):  # noqa: F811 - imported pytest fixture
    calls = []

    class Runtime:
        async def answer(self, *args, **kwargs):
            calls.append(kwargs)
            return AnswerResponse(generated_text="answer", citations=[], sources=[], retrieval_count=0, elapsed_ms=0)

    async def scenario():
        bridge.configure(Runtime())
        scope = Scope(tenant_id="t", owner_id="a")
        with bridge.bind_request(scope, "s", source_session_ids=("past",)):
            assert bridge.require_context().source_read_scope == "quotes"
            with bridge.bind_request(scope, "s", source_session_ids=("past",), source_read_scope="events"):
                assert bridge.require_context().source_read_scope == "events"
                await bridge.engine_plugin.generate("which recipe")
            assert bridge.require_context().source_read_scope == "quotes"
            await bridge.engine_plugin.generate("which recipe")
        with pytest.raises(PermissionError):
            bridge.require_context()
        assert calls[0]["source_read_scope"] == "events"
        assert "source_read_scope" not in calls[1]
        assert all(call["source_session_ids"] == ("past",) for call in calls)
        with pytest.raises(ValueError, match="source_read_scope"), bridge.bind_request(
            scope, "s", source_read_scope="all"
        ):
            pass

    asyncio.run(scenario())
