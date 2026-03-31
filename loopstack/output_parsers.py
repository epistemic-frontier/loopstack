from __future__ import annotations

import json
from typing import Any


def parse_text(raw: str) -> str:
    if not isinstance(raw, str):
        raise ValueError(f"Expected text output, got {type(raw).__name__}.")
    return raw.strip()


def validate_nonempty_text(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Expected non-empty text output.")
    return value.strip()


def parse_json_output(raw: str) -> Any:
    if not isinstance(raw, str):
        raise ValueError(f"Expected JSON text output, got {type(raw).__name__}.")
    text = raw.strip()
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.startswith("```")]
        text = "\n".join(lines).strip()
    return json.loads(text)
