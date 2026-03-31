from __future__ import annotations

from pathlib import Path

from .compiler_codegen import compile_with_llm
from .taskfile import TaskDefinition, load_task_file


def compile_task_file(
    task_path: str | Path = "TASK.md",
    *,
    output_path: str | Path = "loopstack.py",
) -> TaskDefinition:
    task = load_task_file(task_path)
    repo_root = Path(task_path).resolve().parent
    task_source = Path(task_path).read_text()
    Path(output_path).write_text(render_compiled_program(task, repo_root=repo_root, task_source=task_source))
    return task


def render_compiled_program(
    task: TaskDefinition,
    *,
    repo_root: str | Path,
    task_source: str | None = None,
) -> str:
    resolved_root = Path(repo_root).resolve()
    task_text = task_source if task_source is not None else (resolved_root / "TASK.md").read_text()
    return compile_with_llm(task, repo_root=resolved_root, task_source=task_text)
