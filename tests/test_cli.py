from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from loopstack.cli import main


def test_cli_init_compile_and_run(tmp_path: Path) -> None:
    repo_root = tmp_path

    assert main(["init", "--repo-root", str(repo_root)]) == 0
    task_path = repo_root / "TASK.md"
    task_path.write_text(
        """---
task_type: coding
max_iterations: 2
write_scope:
  - "src/**"
verifiers:
  - "python -c \\"print('ok')\\""
memory: true
---

Check repository health.
"""
    )

    assert main(["compile", "--repo-root", str(repo_root)]) == 0
    assert (repo_root / "loopstack.py").exists()
    assert main(["run", "--repo-root", str(repo_root), "--python", sys.executable]) == 0

    state_path = repo_root / ".loopstack" / "state.json"
    memory_path = repo_root / ".loopstack" / "memory.json"
    runs_dir = repo_root / ".loopstack" / "runs"

    assert state_path.exists()
    assert memory_path.exists()
    assert any(runs_dir.iterdir())


def test_compiled_program_exits_nonzero_when_verifier_fails(tmp_path: Path) -> None:
    repo_root = tmp_path
    (repo_root / "TASK.md").write_text(
        """---
task_type: coding
max_iterations: 1
write_scope:
  - "src/**"
verifiers:
  - "python -c \\"import sys; sys.exit(3)\\""
memory: false
---

Fail a verifier.
"""
    )

    assert main(["compile", "--repo-root", str(repo_root)]) == 0
    result = subprocess.run(
        [sys.executable, str(repo_root / "loopstack.py")],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1


def test_compiled_program_preserves_nested_quotes_in_verifiers(tmp_path: Path) -> None:
    repo_root = tmp_path
    (repo_root / "TASK.md").write_text(
        """---
task_type: coding
max_iterations: 1
write_scope:
  - "src/**"
verifiers:
  - 'python -c "from pathlib import Path; root = Path(\\".\\"); required = [\\"TASK.md\\"]; missing = [name for name in required if not (root / name).exists()]; raise SystemExit(1 if missing else 0)"'
memory: true
---

Preserve nested quotes in compiled verifiers.
"""
    )

    assert main(["compile", "--repo-root", str(repo_root)]) == 0
    assert main(["run", "--repo-root", str(repo_root), "--python", sys.executable]) == 0
