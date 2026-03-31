from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from .agent_runtime import ChatCompletionClient, CompletionClient, RoleRunner
from .agent_trace import JsonDict, emit_trace
from .autoresearch_runtime import run_autoresearch_task
from .output_parsers import parse_json_output, parse_text, validate_nonempty_text
from .runtime import VerifierResult, _finalize_run, _run_verifiers, run_compiled_task as _run_compiled_task
from .taskfile import TaskDefinition
from .tool_environment import ToolCatalog, ToolHandler, ToolHost, ToolInvocation, ToolSpec, ToolTurn
from .workflow_spec import CommitSpec, OutputSpec, PhaseSpec, RoleSpec


@dataclass(frozen=True)
class RunSession:
    run_id: str
    repo_root: Path
    runtime_dir: Path
    runs_dir: Path
    trace: list[JsonDict]


def begin_run(*, repo_root: str | Path, task: Mapping[str, Any]) -> RunSession:
    resolved_root = Path(repo_root).resolve()
    runtime_dir = resolved_root / ".loopstack"
    runs_dir = runtime_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    trace: list[JsonDict] = []
    emit_trace(trace, "load", run_id=run_id, repo_root=str(resolved_root), task=dict(task))
    emit_trace(trace, "propose", summary=str(task.get("body", "")), max_iterations=int(task.get("max_iterations", 0)))
    emit_trace(
        trace,
        "materialize",
        write_scope=list(task.get("write_scope", [])),
        verifier_count=len(tuple(task.get("verifiers", ()))),
    )
    return RunSession(
        run_id=run_id,
        repo_root=resolved_root,
        runtime_dir=runtime_dir,
        runs_dir=runs_dir,
        trace=trace,
    )


def finalize_run(
    session: RunSession,
    *,
    task: Mapping[str, Any],
    verifiers: list[VerifierResult],
    extra_state: Mapping[str, Any] | None,
    memory_payload: dict[str, Any] | None,
) -> int:
    task_definition = TaskDefinition.from_mapping(task, body=str(task["body"]))
    decision = "accept" if all(item.passed for item in verifiers) else "revise"
    emit_trace(session.trace, "decide", decision=decision, verifier_results=[item.to_dict() for item in verifiers])
    return _finalize_run(
        run_id=session.run_id,
        task=task_definition,
        runtime_dir=session.runtime_dir,
        runs_dir=session.runs_dir,
        trace=session.trace,
        verifier_results=verifiers,
        extra_state=extra_state,
        memory_payload=memory_payload,
    )


def run_verifier_suite(commands: tuple[str, ...] | list[str], *, repo_root: str | Path, trace: list[JsonDict]) -> list[VerifierResult]:
    return _run_verifiers(tuple(commands), repo_root=Path(repo_root).resolve(), trace=trace)


def run_autoresearch_loop(
    task: Mapping[str, Any],
    *,
    repo_root: str | Path,
    runtime_dir: str | Path,
    trace: list[JsonDict],
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    task_definition = TaskDefinition.from_mapping(task, body=str(task["body"]))
    return run_autoresearch_task(
        task_definition,
        repo_root=Path(repo_root).resolve(),
        runtime_dir=Path(runtime_dir).resolve(),
        trace=trace,
    )


def run_compiled_task(task: Mapping[str, Any], *, repo_root: str | Path) -> int:
    return _run_compiled_task(task, repo_root=repo_root)


__all__ = [
    "ChatCompletionClient",
    "CommitSpec",
    "CompletionClient",
    "JsonDict",
    "OutputSpec",
    "PhaseSpec",
    "RoleRunner",
    "RoleSpec",
    "RunSession",
    "ToolCatalog",
    "ToolHandler",
    "ToolHost",
    "ToolInvocation",
    "ToolSpec",
    "ToolTurn",
    "VerifierResult",
    "begin_run",
    "emit_trace",
    "finalize_run",
    "parse_json_output",
    "parse_text",
    "run_autoresearch_loop",
    "run_compiled_task",
    "run_verifier_suite",
    "validate_nonempty_text",
]
