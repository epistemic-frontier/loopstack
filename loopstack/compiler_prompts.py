from __future__ import annotations

from pathlib import Path

from .taskfile import TaskDefinition

STDLIB_CONTRACT = """
Use only Python standard library imports plus loopstack.stdlib.

loopstack.stdlib exposes:
- begin_run(repo_root, task) -> RunSession
- finalize_run(session, task, verifiers, extra_state, memory_payload) -> int
- run_verifier_suite(commands, repo_root, trace) -> list[VerifierResult]
- run_autoresearch_loop(task, repo_root, runtime_dir, trace) -> (status, summary, memory_payload)
- run_compiled_task(task, repo_root) -> int
- emit_trace(trace, event, **payload)

For autoresearch tasks, generate code that:
- embeds TASK_PAYLOAD
- begins a run
- executes run_autoresearch_loop(...)
- runs final verifiers
- finalizes the run

For non-autoresearch tasks, generate code that:
- embeds TASK_PAYLOAD
- calls loopstack.stdlib.run_compiled_task(...)

The generated file must define main() and use:
if __name__ == "__main__":
    raise SystemExit(main())

The generated file name is loopstack.py, so it must avoid importing loopstack.stdlib directly before removing the current directory from sys.path.
Use importlib.import_module("loopstack.stdlib") after adjusting sys.path.
Return Python source only.
""".strip()


def build_compile_system_prompt() -> str:
    return "\n\n".join(
        (
            "You are a compiler that turns TASK.md into a runnable task-specific loopstack.py program.",
            STDLIB_CONTRACT,
        )
    )


def build_compile_user_prompt(
    task: TaskDefinition,
    *,
    repo_root: Path,
    task_source: str,
    context_blocks: list[str],
) -> str:
    return "\n\n".join(
        (
            f"Repository root: {repo_root}",
            "TASK.md source:",
            task_source,
            "Task payload as JSON-like data:",
            str(task.to_dict()),
            "Context files:",
            "\n\n".join(context_blocks) if context_blocks else "(none)",
            "Generate the final loopstack.py now.",
        )
    )
