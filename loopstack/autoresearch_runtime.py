from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .agent_runtime import ChatCompletionClient, CompletionClient, RoleRunner
from .agent_trace import JsonDict, copy_value, emit_trace
from .output_parsers import parse_json_output, parse_text, validate_nonempty_text
from .taskfile import TaskDefinition
from .tool_environment import ToolHost
from .workflow_spec import CommitSpec, OutputSpec, PhaseSpec, RoleSpec


@dataclass(frozen=True)
class CommandResult:
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


@dataclass(frozen=True)
class AutoresearchConfig:
    prepare_command: str
    train_command: str
    evaluate_command: str
    metric_name: str
    metric_direction: str
    tie_breaker_metric: str
    stagnation_limit: int
    oracle_model: str | None
    oracle_base_url: str | None

    @classmethod
    def from_task(cls, task: TaskDefinition) -> AutoresearchConfig:
        metadata = task.metadata
        prepare_command = _read_required_text(metadata, "prepare_command")
        train_command = _read_required_text(metadata, "train_command")
        evaluate_command = _read_required_text(metadata, "evaluate_command")
        metric_name = str(metadata.get("metric_name", "val_accuracy")).strip() or "val_accuracy"
        metric_direction = str(metadata.get("metric_direction", "maximize")).strip() or "maximize"
        tie_breaker_metric = str(metadata.get("tie_breaker_metric", "val_loss")).strip() or "val_loss"
        stagnation_limit = int(metadata.get("stagnation_limit", 2))
        oracle_model = _read_optional_text(metadata.get("oracle_model"))
        oracle_base_url = _read_optional_text(metadata.get("oracle_base_url"))
        return cls(
            prepare_command=prepare_command,
            train_command=train_command,
            evaluate_command=evaluate_command,
            metric_name=metric_name,
            metric_direction=metric_direction,
            tie_breaker_metric=tie_breaker_metric,
            stagnation_limit=stagnation_limit,
            oracle_model=oracle_model,
            oracle_base_url=oracle_base_url,
        )


def run_autoresearch_task(
    task: TaskDefinition,
    *,
    repo_root: Path,
    runtime_dir: Path,
    trace: list[JsonDict],
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    config = AutoresearchConfig.from_task(task)
    train_path = repo_root / "train.py"
    prepare_path = repo_root / "prepare.py"
    evaluate_path = repo_root / "evaluate.py"
    if not train_path.exists() or not prepare_path.exists() or not evaluate_path.exists():
        raise FileNotFoundError("autoresearch task requires prepare.py, train.py, and evaluate.py at the repo root.")

    runs_dir = runtime_dir / "autoresearch"
    candidates_dir = runs_dir / "candidates"
    metrics_dir = runs_dir / "metrics"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    prepare_result = _run_command(
        config.prepare_command,
        repo_root=repo_root,
        trace=trace,
        event_prefix="prepare",
    )
    if not prepare_result.passed:
        raise RuntimeError(f"prepare_command failed:\n{prepare_result.stderr}")

    baseline_model_path = metrics_dir / "baseline_model.json"
    baseline_metrics_path = metrics_dir / "baseline.json"
    baseline = _run_training_and_evaluation(
        train_file=train_path,
        model_file=baseline_model_path,
        metrics_file=baseline_metrics_path,
        incumbent_file=None,
        config=config,
        repo_root=repo_root,
        trace=trace,
    )
    if baseline["metrics"].get("failed"):
        raise RuntimeError("baseline training or evaluation failed")
    baseline_record = {
        "train_file": str(train_path),
        "model_file": str(baseline_model_path),
        "metrics_file": str(baseline_metrics_path),
        **baseline,
    }
    incumbent = baseline_record
    history: list[dict[str, Any]] = []

    client = _build_oracle_client(task)
    runner = RoleRunner(client, ToolHost({}, {}), trace=trace)
    planner_role = _build_planner_role(task)
    implementer_role = _build_implementer_role(task)

    stagnation_rounds = 0
    for iteration in range(task.max_iterations):
        emit_trace(
            trace,
            "iteration_start",
            iteration=iteration + 1,
            incumbent_metrics=copy_value(incumbent["metrics"]),
        )
        state = {
            "task_body": task.body,
            "train_source": train_path.read_text(),
            "prepare_source": prepare_path.read_text(),
            "evaluate_source": evaluate_path.read_text(),
            "incumbent_metrics": json.dumps(incumbent["metrics"], ensure_ascii=False, indent=2),
            "history": json.dumps(history, ensure_ascii=False, indent=2),
            "iteration_label": f"iteration {iteration + 1}",
        }
        proposal_state = runner.run(planner_role, state)
        proposal = proposal_state["proposal"]
        implementation_state = runner.run(
            implementer_role,
            {
                **state,
                "proposal": json.dumps(proposal, ensure_ascii=False, indent=2),
            },
        )
        candidate_source = implementation_state["candidate_train_py"]
        candidate_dir = candidates_dir / f"iteration_{iteration + 1:02d}"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        candidate_path = candidate_dir / "train.py"
        candidate_path.write_text(candidate_source)
        syntax_result = _run_command(
            f"python -m py_compile {candidate_path}",
            repo_root=repo_root,
            trace=trace,
            event_prefix="candidate_syntax",
        )
        candidate_model_path = metrics_dir / f"iteration_{iteration + 1:02d}_model.json"
        candidate_metrics_path = metrics_dir / f"iteration_{iteration + 1:02d}.json"
        if not syntax_result.passed:
            iteration_record = {
                "iteration": iteration + 1,
                "proposal": proposal,
                "decision": "reject",
                "reason": "syntax_error",
                "syntax": syntax_result.to_dict(),
            }
            history.append(iteration_record)
            emit_trace(trace, "iteration_reject", **iteration_record)
            stagnation_rounds += 1
            if stagnation_rounds >= config.stagnation_limit:
                break
            continue
        candidate = _run_training_and_evaluation(
            train_file=candidate_path,
            model_file=candidate_model_path,
            metrics_file=candidate_metrics_path,
            incumbent_file=Path(str(incumbent["metrics_file"])),
            config=config,
            repo_root=repo_root,
            trace=trace,
        )
        decision = "accept" if _is_better(candidate["metrics"], incumbent["metrics"], config) else "reject"
        iteration_record = {
            "iteration": iteration + 1,
            "proposal": proposal,
            "candidate_train_file": str(candidate_path),
            "candidate_model_file": str(candidate_model_path),
            "candidate_metrics_file": str(candidate_metrics_path),
            "decision": decision,
            "metrics": candidate["metrics"],
            "training": candidate["training"],
            "evaluation": candidate["evaluation"],
        }
        if decision == "accept":
            shutil.copyfile(candidate_path, train_path)
            incumbent = {
                "train_file": str(train_path),
                "model_file": str(candidate_model_path),
                "metrics_file": str(candidate_metrics_path),
                **candidate,
            }
            stagnation_rounds = 0
            emit_trace(trace, "iteration_accept", **copy_value(iteration_record))
        else:
            stagnation_rounds += 1
            emit_trace(trace, "iteration_reject", **copy_value(iteration_record))
        history.append(iteration_record)
        if stagnation_rounds >= config.stagnation_limit:
            break

    summary = {
        "mode": "autoresearch",
        "baseline": baseline_record,
        "best": incumbent,
        "iterations": history,
    }
    (runs_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return "passed", summary, {
        "recent_runs": [
            {
                "task_type": task.task_type,
                "best_metrics": incumbent["metrics"],
                "accepted_iterations": sum(1 for item in history if item["decision"] == "accept"),
            }
        ]
    }


def _build_oracle_client(task: TaskDefinition) -> CompletionClient:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    model = _read_optional_text(task.metadata.get("oracle_model")) or os.getenv("AUTORESEARCH_MODEL") or "openai/gpt-4.1-mini"
    base_url = _read_optional_text(task.metadata.get("oracle_base_url")) or os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1/chat/completions"
    if api_key:
        return ChatCompletionClient(base_url=base_url, api_key=api_key, model=model)
    return DeterministicAutoresearchClient()


def _build_planner_role(task: TaskDefinition) -> RoleSpec:
    return RoleSpec(
        name="planner",
        purpose="Propose the next bounded model improvement.",
        system_context=task.body,
        required_inputs=(
            "task_body",
            "train_source",
            "prepare_source",
            "evaluate_source",
            "incumbent_metrics",
            "history",
            "iteration_label",
        ),
        field_descriptions={
            "task_body": "Detailed TASK.md body for the compiled loop.",
            "train_source": "Current train.py source that may be improved.",
            "prepare_source": "Fixed prepare.py contract.",
            "evaluate_source": "Fixed evaluate.py contract.",
            "incumbent_metrics": "Current best metrics in JSON form.",
            "history": "Prior iteration outcomes in JSON form.",
            "iteration_label": "Current iteration label.",
        },
        phases=(
            PhaseSpec(
                name="emit_proposal",
                purpose="Generate one concrete proposal for the next training iteration.",
                dominant_mode="plan",
                reads=(
                    "task_body",
                    "train_source",
                    "prepare_source",
                    "evaluate_source",
                    "incumbent_metrics",
                    "history",
                    "iteration_label",
                ),
                instructions=(
                    "Return exactly one JSON object with keys hypothesis, changes, and expected_signal. "
                    "Keep the proposal concrete, bounded, and focused on improving validation language-model metrics "
                    "without breaking the train.py checkpoint schema or the evaluate.py contract."
                ),
                commit=CommitSpec(
                    writes="proposal",
                    output=OutputSpec(
                        instructions="Emit JSON only.",
                        parser=parse_json_output,
                        validator=_validate_proposal,
                    ),
                ),
            ),
        ),
    )


def _build_implementer_role(task: TaskDefinition) -> RoleSpec:
    return RoleSpec(
        name="implementer",
        purpose="Materialize the proposal as an updated train.py.",
        system_context=task.body,
        required_inputs=("task_body", "train_source", "proposal", "incumbent_metrics", "history"),
        field_descriptions={
            "task_body": "Detailed TASK.md body for the compiled loop.",
            "train_source": "Current train.py source.",
            "proposal": "Planner proposal as JSON text.",
            "incumbent_metrics": "Current best metrics as JSON text.",
            "history": "Prior iteration outcomes as JSON text.",
        },
        phases=(
            PhaseSpec(
                name="emit_train_py",
                purpose="Produce a full replacement train.py implementing the proposal.",
                dominant_mode="action",
                reads=("task_body", "train_source", "proposal", "incumbent_metrics", "history"),
                instructions=(
                    "Return the full train.py source only. Preserve the CLI shape and keep train.py responsible "
                    "for writing a model checkpoint that evaluate.py can score."
                ),
                commit=CommitSpec(
                    writes="candidate_train_py",
                    output=OutputSpec(
                        instructions="Emit raw Python source only.",
                        parser=parse_text,
                        validator=validate_nonempty_text,
                    ),
                ),
            ),
        ),
    )


class DeterministicAutoresearchClient:
    def complete(self, messages: list[JsonDict], stop: list[str] | None = None) -> str:
        system_text = str(messages[0]["content"])
        user_text = str(messages[-1]["content"])
        if "Current phase: emit_proposal" in system_text:
            proposal = _build_deterministic_proposal(user_text)
            return json.dumps(proposal, ensure_ascii=False, indent=2)
        train_source = _extract_binding(user_text, "train_source")
        return _improve_train_source(train_source)


def _extract_binding(prompt: str, field_name: str) -> str:
    marker = f"[{field_name}]"
    start = prompt.find(marker)
    if start < 0:
        return ""
    start = prompt.find("\n", start)
    if start < 0:
        return ""
    next_binding = prompt.find("\n[", start + 1)
    if next_binding < 0:
        next_binding = len(prompt)
    return prompt[start:next_binding].strip()


def _improve_train_source(source: str) -> str:
    updated = source
    if "n_layer:" in source and "n_embd:" in source:
        updated = re.sub(r"steps: int = \d+", "steps: int = 50", updated, count=1)
        updated = re.sub(r"learning_rate: float = [0-9.]+", "learning_rate: float = 0.0025", updated, count=1)
        updated = re.sub(r"n_layer: int = \d+", "n_layer: int = 2", updated, count=1)
        updated = re.sub(r"n_embd: int = \d+", "n_embd: int = 48", updated, count=1)
        updated = re.sub(r'--steps", type=int, default=\d+', '--steps", type=int, default=50', updated, count=1)
        updated = re.sub(r'--learning-rate", type=float, default=[0-9.]+', '--learning-rate", type=float, default=0.0025', updated, count=1)
        updated = re.sub(r'--n-layer", type=int, default=\d+', '--n-layer", type=int, default=2', updated, count=1)
        updated = re.sub(r'--n-embd", type=int, default=\d+', '--n-embd", type=int, default=48', updated, count=1)
        return updated
    if "order:" in source and "smoothing:" in source:
        updated = re.sub(r"order: int = \d+", "order: int = 2", updated, count=1)
        updated = re.sub(r"smoothing: float = [0-9.]+", "smoothing: float = 0.01", updated, count=1)
        updated = re.sub(r'--order", type=int, default=\d+', '--order", type=int, default=2', updated, count=1)
        updated = re.sub(r'--smoothing", type=float, default=[0-9.]+', '--smoothing", type=float, default=0.01', updated, count=1)
        return updated
    return updated


def _build_deterministic_proposal(user_text: str) -> dict[str, Any]:
    train_source = _extract_binding(user_text, "train_source")
    if "n_layer:" in train_source and "n_embd:" in train_source:
        return {
            "hypothesis": "A slightly deeper and wider transformer trained for longer should lower val_bpb while preserving the GPT-style training setup.",
            "changes": [
                "Increase transformer depth from 1 layer to 2 layers.",
                "Increase embedding width from 32 to 48.",
                "Train for 50 steps with a slightly smaller learning rate of 0.0025.",
            ],
            "expected_signal": "Lower val_bpb on validation without changing the checkpoint contract.",
        }
    return {
        "hypothesis": "Use a smoothed bigram language model with a smaller additive prior so val_bpb drops on the held-out split.",
        "changes": [
            "Raise the default n-gram order from 1 to 2.",
            "Lower the default additive smoothing value to 0.01.",
            "Keep the checkpoint and CLI contract unchanged.",
        ],
        "expected_signal": "Lower val_bpb on validation without breaking checkpoint compatibility.",
    }


def _validate_proposal(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Proposal must be a JSON object.")
    for key in ("hypothesis", "changes", "expected_signal"):
        if key not in value:
            raise ValueError(f"Proposal is missing required key: {key}.")
    if not isinstance(value["hypothesis"], str) or not value["hypothesis"].strip():
        raise ValueError("Proposal hypothesis must be non-empty.")
    if not isinstance(value["expected_signal"], str) or not value["expected_signal"].strip():
        raise ValueError("Proposal expected_signal must be non-empty.")
    if not isinstance(value["changes"], list) or not value["changes"]:
        raise ValueError("Proposal changes must be a non-empty list.")
    cleaned_changes = [str(item).strip() for item in value["changes"] if str(item).strip()]
    if not cleaned_changes:
        raise ValueError("Proposal changes must contain at least one non-empty item.")
    return {
        "hypothesis": value["hypothesis"].strip(),
        "changes": cleaned_changes,
        "expected_signal": value["expected_signal"].strip(),
    }


def _run_training_and_evaluation(
    *,
    train_file: Path,
    model_file: Path,
    metrics_file: Path,
    incumbent_file: Path | None,
    config: AutoresearchConfig,
    repo_root: Path,
    trace: list[JsonDict],
) -> dict[str, Any]:
    training_command = _format_command(
        config.train_command,
        train_file=train_file,
        model_file=model_file,
        metrics_file=metrics_file,
        incumbent_file=incumbent_file,
    )
    training = _run_command(training_command, repo_root=repo_root, trace=trace, event_prefix="training")
    if not training.passed:
        return {"metrics": {"failed": True}, "training": training.to_dict(), "evaluation": None}
    evaluation_command = _format_command(
        config.evaluate_command,
        train_file=train_file,
        model_file=model_file,
        metrics_file=metrics_file,
        incumbent_file=incumbent_file,
    )
    evaluation = _run_command(evaluation_command, repo_root=repo_root, trace=trace, event_prefix="evaluation")
    if not evaluation.passed:
        return {
            "metrics": {"failed": True},
            "training": training.to_dict(),
            "evaluation": evaluation.to_dict(),
        }
    metrics = json.loads(metrics_file.read_text())
    return {
        "metrics": metrics,
        "training": training.to_dict(),
        "evaluation": evaluation.to_dict(),
    }


def _format_command(
    command: str,
    *,
    train_file: Path,
    model_file: Path,
    metrics_file: Path,
    incumbent_file: Path | None,
) -> str:
    incumbent_text = "" if incumbent_file is None else str(incumbent_file)
    incumbent_args = "" if incumbent_file is None else f"--incumbent {incumbent_text}"
    return command.format(
        train_file=str(train_file),
        model_file=str(model_file),
        metrics_file=str(metrics_file),
        incumbent_file=incumbent_text,
        incumbent_args=incumbent_args,
    )


def _run_command(
    command: str,
    *,
    repo_root: Path,
    trace: list[JsonDict],
    event_prefix: str,
) -> CommandResult:
    emit_trace(trace, f"{event_prefix}_start", command=command)
    completed = subprocess.run(
        command,
        cwd=repo_root,
        shell=True,
        executable="/bin/zsh",
        capture_output=True,
        text=True,
    )
    result = CommandResult(
        command=command,
        exit_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
    emit_trace(trace, f"{event_prefix}_result", **result.to_dict())
    return result


def _is_better(candidate: dict[str, Any], incumbent: dict[str, Any], config: AutoresearchConfig) -> bool:
    if candidate.get("failed"):
        return False
    candidate_primary = float(candidate[config.metric_name])
    incumbent_primary = float(incumbent[config.metric_name])
    if candidate_primary != incumbent_primary:
        if config.metric_direction == "minimize":
            return candidate_primary < incumbent_primary
        return candidate_primary > incumbent_primary
    candidate_tie = float(candidate[config.tie_breaker_metric])
    incumbent_tie = float(incumbent[config.tie_breaker_metric])
    return candidate_tie < incumbent_tie


def _read_required_text(metadata: dict[str, Any] | Any, key: str) -> str:
    if not isinstance(metadata, dict):
        raise ValueError("Task metadata must be a mapping.")
    value = _read_optional_text(metadata.get(key))
    if value is None:
        raise ValueError(f"autoresearch task is missing required metadata: {key}")
    return value


def _read_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
