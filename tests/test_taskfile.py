from __future__ import annotations

from pathlib import Path

from loopstack.taskfile import load_task_file


def test_load_task_file_parses_front_matter_and_body(tmp_path: Path) -> None:
    task_path = tmp_path / "TASK.md"
    task_path.write_text(
        """---
task_type: coding
max_iterations: 5
write_scope:
  - "src/**"
verifiers:
  - "pytest -q"
memory: false
---

Fix the failing tests.
"""
    )

    task = load_task_file(task_path)

    assert task.task_type == "coding"
    assert task.max_iterations == 5
    assert task.write_scope == ("src/**",)
    assert task.verifiers == ("pytest -q",)
    assert task.memory is False
    assert task.body == "Fix the failing tests."


def test_load_task_file_preserves_additional_metadata(tmp_path: Path) -> None:
    task_path = tmp_path / "TASK.md"
    task_path.write_text(
        """---
task_type: autoresearch
max_iterations: 3
write_scope:
  - "train.py"
verifiers:
  - "pytest -q"
memory: true
prepare_command: "python prepare.py"
train_command: "python {train_file} --output {metrics_file}"
evaluate_command: "python evaluate.py --summary {metrics_file} --incumbent {incumbent_file}"
metric_name: "val_accuracy"
---

Run the compiled loop.
"""
    )

    task = load_task_file(task_path)

    assert task.metadata["prepare_command"] == "python prepare.py"
    assert task.metadata["metric_name"] == "val_accuracy"
