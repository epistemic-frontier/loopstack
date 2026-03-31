from __future__ import annotations

import ast
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .agent_runtime import ChatCompletionClient
from .compiler_prompts import build_compile_system_prompt, build_compile_user_prompt
from .taskfile import TaskDefinition


class CompilerClient(Protocol):
    def complete(self, messages: list[dict[str, str]], stop: list[str] | None = None) -> str: ...


@dataclass(frozen=True)
class CompilerConfig:
    model: str | None
    base_url: str | None
    api_key_env: str
    context_files: tuple[str, ...]
    max_retries: int

    @classmethod
    def from_task(cls, task: TaskDefinition) -> CompilerConfig:
        metadata = task.metadata
        context_files = tuple(
            str(item).strip()
            for item in metadata.get("compile_context_files", ["TASK.md", "prepare.py", "train.py", "evaluate.py"])
        )
        max_retries = int(metadata.get("compile_max_retries", 2))
        return cls(
            model=_read_optional_text(metadata.get("compile_model")),
            base_url=_read_optional_text(metadata.get("compile_base_url")),
            api_key_env=_read_optional_text(metadata.get("compile_api_key_env")) or "OPENROUTER_API_KEY",
            context_files=tuple(item for item in context_files if item),
            max_retries=max_retries,
        )


def compile_with_llm(
    task: TaskDefinition,
    *,
    repo_root: Path,
    task_source: str,
) -> str:
    config = CompilerConfig.from_task(task)
    client = _build_compiler_client(task, config=config)
    context_blocks = _load_context_blocks(repo_root, config.context_files)
    system_prompt = build_compile_system_prompt()
    user_prompt = build_compile_user_prompt(task, repo_root=repo_root, task_source=task_source, context_blocks=context_blocks)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    raw = client.complete(messages)
    code = _extract_python_source(raw)
    _validate_compiled_program(code)
    return code


def _build_compiler_client(task: TaskDefinition, *, config: CompilerConfig) -> CompilerClient:
    api_key = os.getenv(config.api_key_env, "").strip()
    if api_key and config.model and config.base_url:
        return ChatCompletionClient(base_url=config.base_url, api_key=api_key, model=config.model)
    return DeterministicCompilerClient(task)


def _load_context_blocks(repo_root: Path, context_files: tuple[str, ...]) -> list[str]:
    blocks: list[str] = []
    for relative_path in context_files:
        path = (repo_root / relative_path).resolve()
        if not path.exists():
            continue
        blocks.append(f"## {relative_path}\n\n{path.read_text()}")
    return blocks


def _extract_python_source(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        filtered = [line for line in lines if not line.startswith("```")]
        return "\n".join(filtered).strip() + "\n"
    return text + ("\n" if not text.endswith("\n") else "")


def _validate_compiled_program(source: str) -> None:
    ast.parse(source)
    if 'import_module("loopstack.stdlib")' not in source:
        raise ValueError("Compiled loopstack.py must import loopstack.stdlib via importlib.")
    if "def main()" not in source:
        raise ValueError("Compiled loopstack.py must define main().")


class DeterministicCompilerClient:
    def __init__(self, task: TaskDefinition) -> None:
        self.task = task

    def complete(self, messages: list[dict[str, str]], stop: list[str] | None = None) -> str:
        payload = json.dumps(self.task.to_dict(), ensure_ascii=False, indent=2)
        if self.task.task_type == "autoresearch":
            return _render_autoresearch_program(payload)
        return _render_generic_program(payload)


def _render_generic_program(payload: str) -> str:
    return f"""from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [path for path in sys.path if Path(path or ".").resolve() != SCRIPT_DIR]
stdlib = importlib.import_module("loopstack.stdlib")

TASK_PAYLOAD = json.loads(
    {payload!r}
)


def main() -> int:
    return stdlib.run_compiled_task(TASK_PAYLOAD, repo_root=SCRIPT_DIR)


if __name__ == "__main__":
    raise SystemExit(main())
"""


def _render_autoresearch_program(payload: str) -> str:
    return f"""from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [path for path in sys.path if Path(path or ".").resolve() != SCRIPT_DIR]
stdlib = importlib.import_module("loopstack.stdlib")

TASK_PAYLOAD = json.loads(
    {payload!r}
)


def main() -> int:
    session = stdlib.begin_run(repo_root=SCRIPT_DIR, task=TASK_PAYLOAD)
    status, summary, memory_payload = stdlib.run_autoresearch_loop(
        TASK_PAYLOAD,
        repo_root=SCRIPT_DIR,
        runtime_dir=session.runtime_dir,
        trace=session.trace,
    )
    verifiers = stdlib.run_verifier_suite(TASK_PAYLOAD["verifiers"], repo_root=SCRIPT_DIR, trace=session.trace)
    if status != "passed":
        for item in verifiers:
            if item.passed:
                continue
            break
    return stdlib.finalize_run(
        session,
        task=TASK_PAYLOAD,
        verifiers=verifiers,
        extra_state=summary,
        memory_payload=memory_payload if TASK_PAYLOAD.get("memory", True) else None,
    )


if __name__ == "__main__":
    raise SystemExit(main())
"""


def _read_optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
