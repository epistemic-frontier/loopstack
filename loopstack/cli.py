from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from .compiler import compile_task_file
from .taskfile import render_task_template


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Loopstack command line interface.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Create a minimal TASK.md template.")
    init_parser.add_argument("--repo-root", default=".", help="Target repository root.")
    init_parser.add_argument("--force", action="store_true", help="Overwrite TASK.md if it exists.")

    compile_parser = subparsers.add_parser("compile", help="Compile TASK.md into loopstack.py.")
    compile_parser.add_argument("--repo-root", default=".", help="Target repository root.")
    compile_parser.add_argument("--task-file", default="TASK.md", help="Path to the task source file.")
    compile_parser.add_argument("--output", default="loopstack.py", help="Path to the compiled runtime.")

    run_parser = subparsers.add_parser("run", help="Run the compiled loopstack.py program.")
    run_parser.add_argument("--repo-root", default=".", help="Target repository root.")
    run_parser.add_argument("--python", default=sys.executable, help="Python interpreter used to run loopstack.py.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    repo_root = Path(args.repo_root).resolve()
    if args.command == "init":
        return _run_init(repo_root=repo_root, force=args.force)
    if args.command == "compile":
        task_path = repo_root / args.task_file
        output_path = repo_root / args.output
        compile_task_file(task_path, output_path=output_path)
        print(f"Wrote {output_path}", flush=True)
        return 0
    compiled_path = repo_root / "loopstack.py"
    if not compiled_path.exists():
        raise FileNotFoundError(f"{compiled_path} does not exist. Run `loopstack compile` first.")
    completed = subprocess.run([args.python, str(compiled_path)], cwd=repo_root, check=False)
    return completed.returncode


def _run_init(*, repo_root: Path, force: bool) -> int:
    task_path = repo_root / "TASK.md"
    if task_path.exists() and not force:
        raise FileExistsError(f"{task_path} already exists. Use --force to overwrite it.")
    task_path.write_text(render_task_template())
    print(f"Wrote {task_path}", flush=True)
    return 0
