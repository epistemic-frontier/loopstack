from __future__ import annotations

import io
import json
from email.message import Message
from typing import cast
from urllib import error

import pytest

from loopstack.agent_runtime import ChatCompletionClient, CompletionClient, RoleRunner
from loopstack.tool_environment import ToolHost, ToolSpec
from loopstack.workflow_spec import CommitSpec, OutputSpec, PhaseSpec, RoleSpec


class FakeClient:
    def __init__(self, responses: list[str]) -> None:
        self._responses = iter(responses)

    def complete(self, messages: list[dict[str, str]], stop: list[str] | None = None) -> str:
        return next(self._responses)


def test_role_runner_executes_tool_round_trip() -> None:
    trace: list[dict[str, object]] = []
    tool_host = ToolHost(
        {"read_note": ToolSpec(signature="read_note(node_id: str) -> str", description="Read note.")},
        {"read_note": lambda node_id: f"note:{node_id}"},
        trace=trace,
    )
    role = RoleSpec(
        name="planner",
        purpose="Collect notes.",
        system_context="ctx",
        required_inputs=("snapshot",),
        field_descriptions={"snapshot": "Snapshot."},
        phases=(
            PhaseSpec(
                name="investigate",
                purpose="Inspect one node.",
                dominant_mode="observe",
                reads=("snapshot",),
                instructions="Use the tool if needed.",
                commit=CommitSpec(
                    writes="notes",
                    output=OutputSpec(
                        instructions="Return text.",
                        parser=lambda raw: raw.strip(),
                    ),
                ),
                allow_tools=True,
                allowed_tools=("read_note",),
            ),
        ),
    )
    client = FakeClient(['<tool>read_note("node-1")</tool>', "grounded note"])
    runner = RoleRunner(cast(CompletionClient, client), tool_host, trace=trace)

    state = runner.run(role, {"snapshot": "seed"})

    assert state["notes"] == "grounded note"
    assert trace[2]["argument"] == "node-1"


class FakeResponse:
    def __init__(self, payload: object) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None


def test_chat_completion_client_retries_retryable_http_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    client = ChatCompletionClient("https://example.test/v1/chat/completions", "secret", "model")
    attempts: list[int] = []
    sleep_calls: list[float] = []

    def fake_urlopen(http_request: object, timeout: int) -> FakeResponse:
        attempts.append(timeout)
        if len(attempts) == 1:
            raise error.HTTPError(
                client.base_url,
                502,
                "Bad Gateway",
                hdrs=Message(),
                fp=io.BytesIO(b'{"error":"temporary"}'),
            )
        return FakeResponse({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr("loopstack.agent_runtime.request.urlopen", fake_urlopen)
    monkeypatch.setattr("loopstack.agent_runtime.time.sleep", sleep_calls.append)

    assert client.complete([{"role": "user", "content": "hello"}]) == "ok"
    assert attempts == [300, 300]
    assert sleep_calls == [0.5]


def test_chat_completion_client_rejects_empty_choices(monkeypatch: pytest.MonkeyPatch) -> None:
    client = ChatCompletionClient("https://example.test/v1/chat/completions", "secret", "model")

    def fake_urlopen(http_request: object, timeout: int) -> FakeResponse:
        return FakeResponse({"choices": []})

    monkeypatch.setattr("loopstack.agent_runtime.request.urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="contained no choices"):
        client.complete([{"role": "user", "content": "hello"}])


def test_chat_completion_client_rejects_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    client = ChatCompletionClient("https://example.test/v1/chat/completions", "secret", "model")

    class InvalidJsonResponse:
        def read(self) -> bytes:
            return b"{not-json"

        def __enter__(self) -> InvalidJsonResponse:
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

    def fake_urlopen(http_request: object, timeout: int) -> InvalidJsonResponse:
        return InvalidJsonResponse()

    monkeypatch.setattr("loopstack.agent_runtime.request.urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="invalid JSON"):
        client.complete([{"role": "user", "content": "hello"}])


def test_chat_completion_client_accepts_text_choice_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    client = ChatCompletionClient("https://example.test/v1/chat/completions", "secret", "model")

    def fake_urlopen(http_request: object, timeout: int) -> FakeResponse:
        return FakeResponse({"choices": [{"text": "fallback"}]})

    monkeypatch.setattr("loopstack.agent_runtime.request.urlopen", fake_urlopen)

    assert client.complete([{"role": "user", "content": "hello"}]) == "fallback"
