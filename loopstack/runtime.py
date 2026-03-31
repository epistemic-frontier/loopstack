from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from .autoresearch_runtime import run_autoresearch_task
from .agent_trace import JsonDict, emit_trace, trace_to_json
from .taskfile import TaskDefinition


@dataclass(frozen=True)
class VerifierResult:
    command: str
    exit_code: int
    stdout: str
    stderr: str

    @property
    def passed(self) -> bool:
        return self.exit_code == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "exit_code": self.exit_code,
            "passed": self.passed,
            "stdout": self.stdout,
            "stderr": self.stderr,
        }


def run_compiled_task(task_payload: Mapping[str, Any], *, repo_root: str | Path) -> int:
    task = TaskDefinition.from_mapping(task_payload, body=str(task_payload["body"]))
    repo_root = Path(repo_root).resolve()
    runtime_dir = repo_root / ".loopstack"
    runs_dir = runtime_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    trace: list[JsonDict] = []
    emit_trace(trace, "load", run_id=run_id, repo_root=str(repo_root), task=task.to_dict())
    emit_trace(trace, "propose", summary=task.body, max_iterations=task.max_iterations)
    emit_trace(
        trace,
        "materialize",
        write_scope=list(task.write_scope),
        verifier_count=len(task.verifiers),
    )
    if task.task_type == "autoresearch":
        status, state_payload, memory_payload = run_autoresearch_task(
            task,
            repo_root=repo_root,
            runtime_dir=runtime_dir,
            trace=trace,
        )
        verifier_results = _run_verifiers(task.verifiers, repo_root=repo_root, trace=trace)
        all_passed = status == "passed" and all(result.passed for result in verifier_results)
        decision = "accept" if all_passed else "revise"
        emit_trace(trace, "decide", decision=decision, verifier_results=[result.to_dict() for result in verifier_results])
        return _finalize_run(
            run_id=run_id,
            task=task,
            runtime_dir=runtime_dir,
            runs_dir=runs_dir,
            trace=trace,
            verifier_results=verifier_results,
            extra_state=state_payload,
            memory_payload=memory_payload if task.memory else None,
        )
    verifier_results = _run_verifiers(task.verifiers, repo_root=repo_root, trace=trace)
    decision = "accept" if all(result.passed for result in verifier_results) else "revise"
    emit_trace(trace, "decide", decision=decision, verifier_results=[result.to_dict() for result in verifier_results])
    return _finalize_run(
        run_id=run_id,
        task=task,
        runtime_dir=runtime_dir,
        runs_dir=runs_dir,
        trace=trace,
        verifier_results=verifier_results,
        extra_state=None,
        memory_payload=_build_memory_payload(run_id, task, verifier_results) if task.memory else None,
    )


def _run_verifiers(
    commands: tuple[str, ...],
    *,
    repo_root: Path,
    trace: list[JsonDict],
) -> list[VerifierResult]:
    results: list[VerifierResult] = []
    for command in commands:
        emit_trace(trace, "verify_start", command=command)
        completed = subprocess.run(
            command,
            cwd=repo_root,
            shell=True,
            executable="/bin/zsh",
            capture_output=True,
            text=True,
        )
        result = VerifierResult(
            command=command,
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
        emit_trace(trace, "verify_result", **result.to_dict())
        results.append(result)
    return results


def _build_memory_payload(
    run_id: str,
    task: TaskDefinition,
    verifier_results: list[VerifierResult],
) -> dict[str, Any]:
    summary = {
        "run_id": run_id,
        "task_type": task.task_type,
        "body": task.body,
        "failing_verifiers": [result.command for result in verifier_results if not result.passed],
    }
    return {"recent_runs": [summary]}


def _finalize_run(
    *,
    run_id: str,
    task: TaskDefinition,
    runtime_dir: Path,
    runs_dir: Path,
    trace: list[JsonDict],
    verifier_results: list[VerifierResult],
    extra_state: Mapping[str, Any] | None,
    memory_payload: dict[str, Any] | None,
) -> int:
    all_passed = all(result.passed for result in verifier_results)
    memory_path = runtime_dir / "memory.json"
    if memory_payload is not None:
        memory_path.write_text(json.dumps(memory_payload, ensure_ascii=False, indent=2))
        emit_trace(trace, "remember", path=str(memory_path), entries=len(memory_payload["recent_runs"]))
    trace_path = runs_dir / f"{run_id}.json"
    trace_path.write_text(trace_to_json(trace))
    state_payload: dict[str, Any] = {
        "run_id": run_id,
        "status": "passed" if all_passed else "failed",
        "task": task.to_dict(),
        "verifiers": [result.to_dict() for result in verifier_results],
        "trace_file": str(trace_path),
    }
    if extra_state is not None:
        state_payload.update(dict(extra_state))
    (runtime_dir / "state.json").write_text(json.dumps(state_payload, ensure_ascii=False, indent=2))
    print(f"Loopstack run: {run_id}", flush=True)
    print(f"Trace: {trace_path}", flush=True)
    print(f"State: {runtime_dir / 'state.json'}", flush=True)
    if memory_payload is not None:
        print(f"Memory: {memory_path}", flush=True)
    return 0 if all_passed else 1
