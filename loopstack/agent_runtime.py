from __future__ import annotations

import json
import time
from typing import Any, Protocol
from urllib import error, request

from .agent_trace import JsonDict, copy_messages, copy_value, emit_trace
from .tool_environment import ToolHost
from .workflow_spec import PhaseSpec, RoleSpec, StateDict


class CompletionClient(Protocol):
    def complete(self, messages: list[JsonDict], stop: list[str] | None = None) -> str: ...


class ChatCompletionClient:
    _MAX_ATTEMPTS = 3
    _RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
    _BASE_BACKOFF_SECONDS = 0.5

    def __init__(self, base_url: str, api_key: str, model: str, timeout: int = 300) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def complete(self, messages: list[JsonDict], stop: list[str] | None = None) -> str:
        payload: JsonDict = {
            "model": self.model,
            "messages": messages,
        }
        if stop:
            payload["stop"] = stop
        raw_data = self._request_with_retries(payload)
        data = self._load_response_json(raw_data)
        choices = data.get("choices")
        if not isinstance(choices, list):
            raise RuntimeError(f"API response missing choices list from {self.base_url}: {data}")
        if not choices:
            raise RuntimeError(f"API response contained no choices from {self.base_url}: {data}")
        choice = choices[0]
        if not isinstance(choice, dict):
            raise RuntimeError(f"API response returned invalid choice payload from {self.base_url}: {choice!r}")
        message = choice.get("message")
        if not isinstance(message, dict):
            text = choice.get("text")
            if isinstance(text, str):
                return text
            raise RuntimeError(f"API response missing message content from {self.base_url}: {choice!r}")
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            if parts:
                return "".join(parts)
        raise RuntimeError(f"Model returned non-text content from {self.base_url}: {message!r}")

    def _request_with_retries(self, payload: JsonDict) -> str:
        last_error: Exception | None = None
        for attempt in range(1, self._MAX_ATTEMPTS + 1):
            try:
                return self._post(payload)
            except error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                message = self._format_http_error(exc.code, body)
                if attempt >= self._MAX_ATTEMPTS or exc.code not in self._RETRYABLE_STATUS_CODES:
                    raise RuntimeError(message) from exc
                last_error = RuntimeError(message)
            except (error.URLError, TimeoutError, OSError) as exc:
                message = (
                    f"API request to {self.base_url} failed after {attempt} attempt"
                    f"{'' if attempt == 1 else 's'}: {exc}"
                )
                if attempt >= self._MAX_ATTEMPTS:
                    raise RuntimeError(message) from exc
                last_error = RuntimeError(message)
            time.sleep(self._BASE_BACKOFF_SECONDS * (2 ** (attempt - 1)))
        raise RuntimeError(f"API request to {self.base_url} failed: {last_error}") from last_error

    def _post(self, payload: JsonDict) -> str:
        http_request = request.Request(
            self.base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with request.urlopen(http_request, timeout=self.timeout) as response:
            return response.read().decode("utf-8")

    def _load_response_json(self, raw_data: str) -> JsonDict:
        try:
            data = json.loads(raw_data)
        except json.JSONDecodeError as exc:
            preview = raw_data[:200]
            raise RuntimeError(f"API returned invalid JSON from {self.base_url}: {preview!r}") from exc
        if not isinstance(data, dict):
            raise RuntimeError(f"API response must be a JSON object from {self.base_url}: {data!r}")
        return data

    def _format_http_error(self, status_code: int, body: str) -> str:
        body_preview = body[:200]
        return f"API error {status_code} from {self.base_url}: {body_preview}"


class RoleRunner:
    def __init__(
        self,
        client: CompletionClient,
        tool_host: ToolHost,
        trace: list[JsonDict] | None = None,
        debug_mode: bool = False,
    ) -> None:
        self.client = client
        self.tool_host = tool_host
        self.trace = trace
        self.debug_mode = debug_mode

    def run(self, role: RoleSpec, initial_state: StateDict) -> StateDict:
        missing = [name for name in role.required_inputs if name not in initial_state]
        if missing:
            raise ValueError(f"Missing required inputs for role {role.name}: {missing}")
        state = dict(initial_state)
        for phase in role.phases:
            self._validate_phase_inputs(role, phase, state)
            raw_output = self._run_phase(role, phase, state)
            parsed = phase.commit.output.parser(raw_output)
            if phase.commit.output.validator is not None:
                parsed = phase.commit.output.validator(parsed)
            state[phase.commit.writes] = parsed
            emit_trace(
                self.trace,
                "phase_output",
                role=role.name,
                phase=phase.name,
                writes=[phase.commit.writes],
                value=copy_value(parsed),
            )
        return state

    def _validate_phase_inputs(self, role: RoleSpec, phase: PhaseSpec, state: StateDict) -> None:
        missing = [name for name in phase.reads if name not in state]
        if missing:
            raise ValueError(f"Missing phase inputs for {role.name}.{phase.name}: {missing}")

    def _run_phase(self, role: RoleSpec, phase: PhaseSpec, state: StateDict) -> str:
        history: list[JsonDict] = [
            {"role": "system", "content": self._render_system_prompt(role, phase)},
            {"role": "user", "content": self._render_user_prompt(role, phase, state)},
        ]
        tool_rounds = 0
        while True:
            stop_tokens = ["<stop>"] if phase.allow_tools else None
            emit_trace(
                self.trace,
                "oracle_request",
                role=role.name,
                phase=phase.name,
                messages=copy_messages(history),
                stop=list(stop_tokens) if stop_tokens is not None else None,
            )
            raw = self.client.complete(history, stop=stop_tokens)
            emit_trace(self.trace, "oracle_response", role=role.name, phase=phase.name, raw=raw)
            turn = self.tool_host.parse_turn(raw) if phase.allow_tools else None
            if turn is None:
                history.append({"role": "assistant", "content": raw})
                if self.debug_mode:
                    print(f"\n[phase {phase.name} output]\n{raw}\n", flush=True)
                return raw
            if tool_rounds >= phase.max_tool_rounds:
                raise RuntimeError(f"Phase {phase.name} exceeded its tool budget ({phase.max_tool_rounds}).")
            history.append({"role": "assistant", "content": turn.assistant_message})
            invocation = turn.invocation
            tool_result = self.tool_host.execute(
                invocation.name,
                invocation.argument,
                phase.allowed_tools,
                role_name=role.name,
                phase_name=phase.name,
            )
            history.append({"role": "user", "content": f"<result>\n{tool_result}\n</result>"})
            tool_rounds += 1
            if self.debug_mode:
                print(f"\n[tool {invocation.name}({invocation.argument!r})]\n{tool_result}\n", flush=True)

    def _render_system_prompt(self, role: RoleSpec, phase: PhaseSpec) -> str:
        lines = [
            role.system_context.strip(),
            f"Role: {role.name}",
            f"Role purpose: {role.purpose}",
            f"Current phase: {phase.name}",
            f"Phase purpose: {phase.purpose}",
            f"Dominant cognitive mode: {phase.dominant_mode}",
            f"Commit target: {phase.commit.writes}",
            "Phase instructions:",
            phase.instructions.strip(),
        ]
        if role.invariants:
            lines.append("Role invariants:")
            lines.extend(f"- {item}" for item in role.invariants)
        if phase.allow_tools:
            lines.append(self.tool_host.render_protocol(phase.allowed_tools))
        lines.extend(("Output contract:", phase.commit.output.instructions.strip()))
        return "\n".join(lines)

    def _render_user_prompt(self, role: RoleSpec, phase: PhaseSpec, state: StateDict) -> str:
        lines = ["Phase state bindings:"]
        for field_name in phase.reads:
            description = role.field_descriptions.get(field_name, "")
            lines.append(f"\n[{field_name}] {description}" if description else f"\n[{field_name}]")
            lines.append(self._serialize(state[field_name]))
        lines.append("\nEmit the required phase output now.")
        return "\n".join(lines)

    def _serialize(self, value: Any) -> str:
        if isinstance(value, str):
            return value
        return json.dumps(value, indent=2, ensure_ascii=False, default=str)
