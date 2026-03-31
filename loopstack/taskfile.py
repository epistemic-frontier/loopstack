from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

DEFAULT_TASK_TEMPLATE = """---
task_type: coding
max_iterations: 8
write_scope:
  - "src/**"
  - "tests/**"
verifiers:
  - "pytest -q"
memory: true
---

Describe the task here.
"""


@dataclass(frozen=True)
class TaskDefinition:
    task_type: str
    max_iterations: int
    write_scope: tuple[str, ...]
    verifiers: tuple[str, ...]
    memory: bool
    body: str
    metadata: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "task_type": self.task_type,
            "max_iterations": self.max_iterations,
            "write_scope": list(self.write_scope),
            "verifiers": list(self.verifiers),
            "memory": self.memory,
            "body": self.body,
        }
        for key, value in self.metadata.items():
            if key not in payload:
                payload[key] = value
        return payload

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], *, body: str) -> TaskDefinition:
        task_type = str(data.get("task_type", "coding")).strip() or "coding"
        max_iterations = int(data.get("max_iterations", 8))
        write_scope = _read_string_list(data.get("write_scope", ()))
        verifiers = _read_string_list(data.get("verifiers", ()))
        memory = _read_bool(data.get("memory", True))
        cleaned_body = body.strip()
        if not cleaned_body:
            raise ValueError("TASK.md body must not be empty.")
        metadata = {
            key: _copy_value(value)
            for key, value in data.items()
            if key not in {"task_type", "max_iterations", "write_scope", "verifiers", "memory", "body"}
        }
        return cls(
            task_type=task_type,
            max_iterations=max_iterations,
            write_scope=write_scope,
            verifiers=verifiers,
            memory=memory,
            body=cleaned_body,
            metadata=metadata,
        )


def render_task_template() -> str:
    return DEFAULT_TASK_TEMPLATE


def load_task_file(path: str | Path) -> TaskDefinition:
    text = Path(path).read_text()
    front_matter, body = _split_task_document(text)
    payload = _parse_front_matter(front_matter)
    return TaskDefinition.from_mapping(payload, body=body)


def _split_task_document(text: str) -> tuple[list[str], str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("TASK.md must start with a front matter block delimited by ---.")
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            front_matter = lines[1:index]
            body = "\n".join(lines[index + 1 :]).strip()
            return front_matter, body
    raise ValueError("TASK.md front matter is missing its closing --- delimiter.")


def _parse_front_matter(lines: list[str]) -> dict[str, Any]:
    data: dict[str, Any] = {}
    index = 0
    while index < len(lines):
        raw_line = lines[index]
        stripped = raw_line.strip()
        if not stripped:
            index += 1
            continue
        if stripped.startswith("- "):
            raise ValueError(f"Unexpected top-level list item: {raw_line!r}")
        if ":" not in stripped:
            raise ValueError(f"Expected key/value pair in front matter: {raw_line!r}")
        key, raw_value = stripped.split(":", 1)
        key = key.strip()
        value = raw_value.strip()
        if not key:
            raise ValueError("Front matter keys must not be empty.")
        if value:
            data[key] = _parse_scalar(value)
            index += 1
            continue
        items: list[Any] = []
        index += 1
        while index < len(lines):
            item_line = lines[index]
            item_text = item_line.strip()
            if not item_text:
                index += 1
                continue
            if not item_line.startswith("  - "):
                break
            items.append(_parse_scalar(item_text[2:].strip()))
            index += 1
        data[key] = items
    return data


def _parse_scalar(raw_value: str) -> Any:
    text = raw_value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        if text[0] == '"':
            return json.loads(text)
        return text[1:-1].replace("''", "'")
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None
    if text.lstrip("-").isdigit():
        return int(text)
    return text


def _read_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    raise ValueError("memory must be a boolean value.")


def _read_string_list(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list | tuple):
        raise ValueError(f"Expected a list of strings, got {type(value).__name__}.")
    items = [str(item).strip() for item in value]
    if any(not item for item in items):
        raise ValueError("List values must not be empty.")
    return tuple(items)


def _copy_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _copy_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_value(item) for item in value]
    return value
