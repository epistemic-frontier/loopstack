from __future__ import annotations

import json
from pathlib import Path

from loopstack.runtime import run_compiled_task


PREPARE_SOURCE = """from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

CORPUS_FRAGMENTS = (
    "loopstack compiles verifiable task loops into executable runtimes.",
    "autoresearch proposes edits, trains candidates, and validates improvements.",
    "language models improve when context statistics match held out text.",
    "small deterministic corpora keep the demo fast while preserving the workflow.",
)


def ensure_dataset(path: str | Path, *, seed: int = 7, train_repeats: int = 8, val_repeats: int = 3) -> Path:
    generator = random.Random(seed)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    train_text = _build_split(generator, repeats=train_repeats)
    val_text = _build_split(generator, repeats=val_repeats)
    vocab = sorted(set(train_text + val_text))
    token_to_id = {token: index for index, token in enumerate(vocab)}
    payload = {
        "seed": seed,
        "vocab": vocab,
        "train_text": train_text,
        "val_text": val_text,
        "train_ids": [token_to_id[token] for token in train_text],
        "val_ids": [token_to_id[token] for token in val_text],
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return output_path


def _build_split(generator: random.Random, *, repeats: int) -> str:
    fragments = list(CORPUS_FRAGMENTS)
    pieces: list[str] = []
    for _ in range(repeats):
        generator.shuffle(fragments)
        pieces.append(" ".join(fragments))
    return "\\n".join(pieces)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/dataset.json")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    ensure_dataset(args.output, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""

TRAIN_SOURCE = """from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainConfig:
    order: int = 1
    smoothing: float = 1.0


def run_training(dataset_path: str | Path, config: TrainConfig) -> dict[str, object]:
    payload = json.loads(Path(dataset_path).read_text())
    train_ids = [int(token) for token in payload["train_ids"]]
    vocab_size = len(payload["vocab"])
    unigram_counts = [0] * vocab_size
    bigram_counts = [[0] * vocab_size for _ in range(vocab_size)]
    for token in train_ids:
        unigram_counts[token] += 1
    for left, right in zip(train_ids, train_ids[1:]):
        bigram_counts[left][right] += 1
    return {
        "config": asdict(config),
        "vocab": payload["vocab"],
        "train_token_count": len(train_ids),
        "unigram_counts": unigram_counts,
        "bigram_counts": bigram_counts,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/dataset.json")
    parser.add_argument("--order", type=int, default=1)
    parser.add_argument("--smoothing", type=float, default=1.0)
    parser.add_argument("--output", default="artifacts/model.json")
    args = parser.parse_args()
    checkpoint = run_training(
        args.dataset,
        TrainConfig(
            order=args.order,
            smoothing=args.smoothing,
        ),
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""

EVALUATE_SOURCE = """from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def evaluate_checkpoint(dataset_path: str | Path, model_path: str | Path, incumbent_path: str | Path | None, max_val_bpb: float | None) -> None:
    dataset = json.loads(Path(dataset_path).read_text())
    model = json.loads(Path(model_path).read_text())
    train_bpb = _compute_bpb(dataset["train_ids"], model)
    val_bpb = _compute_bpb(dataset["val_ids"], model)
    if max_val_bpb is not None and val_bpb > max_val_bpb:
        raise ValueError("candidate val_bpb is above threshold")
    if incumbent_path is not None and str(incumbent_path).strip():
        incumbent_file = Path(incumbent_path)
        if incumbent_file.exists():
            incumbent = json.loads(incumbent_file.read_text())
            incumbent_val_bpb = float(incumbent["val_bpb"])
            incumbent_train_bpb = float(incumbent["train_bpb"])
            if val_bpb > incumbent_val_bpb:
                raise ValueError("candidate val_bpb regressed")
            if val_bpb == incumbent_val_bpb and train_bpb > incumbent_train_bpb:
                raise ValueError("candidate train_bpb regressed")
    return {"train_bpb": round(train_bpb, 6), "val_bpb": round(val_bpb, 6)}


def _compute_bpb(token_ids: list[int], model: dict[str, object]) -> float:
    unigram_counts = [int(value) for value in model["unigram_counts"]]
    bigram_counts = [[int(item) for item in row] for row in model["bigram_counts"]]
    vocab_size = len(unigram_counts)
    order = int(model["config"]["order"])
    smoothing = float(model["config"]["smoothing"])
    total_unigrams = sum(unigram_counts)
    total_bits = 0.0
    for index, token in enumerate(token_ids):
        if index == 0 or order <= 1:
            probability = (unigram_counts[token] + smoothing) / (total_unigrams + (smoothing * vocab_size))
        else:
            previous = token_ids[index - 1]
            context_total = sum(bigram_counts[previous])
            probability = (bigram_counts[previous][token] + smoothing) / (context_total + (smoothing * vocab_size))
        total_bits += -math.log2(probability)
    return total_bits / len(token_ids)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/dataset.json")
    parser.add_argument("--model", default="artifacts/model.json")
    parser.add_argument("--metrics-output", default="artifacts/metrics.json")
    parser.add_argument("--incumbent", default="")
    parser.add_argument("--max-val-bpb", type=float, default=None)
    args = parser.parse_args()
    metrics = evaluate_checkpoint(args.dataset, args.model, args.incumbent or None, args.max_val_bpb)
    Path(args.metrics_output).write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""

TEST_SOURCE = """from __future__ import annotations

from pathlib import Path

from prepare import ensure_dataset
from train import TrainConfig, run_training


def test_local_contract(tmp_path: Path) -> None:
    dataset_path = ensure_dataset(tmp_path / "data" / "dataset.json", seed=7)
    checkpoint = run_training(dataset_path, TrainConfig(order=2, smoothing=0.2))
    assert checkpoint["config"]["order"] == 2
"""

CONFTEST_SOURCE = """from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
"""


def test_run_compiled_task_executes_autoresearch_loop(tmp_path: Path) -> None:
    repo_root = tmp_path
    (repo_root / "prepare.py").write_text(PREPARE_SOURCE)
    (repo_root / "train.py").write_text(TRAIN_SOURCE)
    (repo_root / "evaluate.py").write_text(EVALUATE_SOURCE)
    tests_dir = repo_root / "tests"
    tests_dir.mkdir()
    (tests_dir / "conftest.py").write_text(CONFTEST_SOURCE)
    (tests_dir / "test_contract.py").write_text(TEST_SOURCE)

    task_payload = {
        "task_type": "autoresearch",
        "max_iterations": 3,
        "write_scope": ["train.py"],
        "verifiers": [
            "python -m py_compile prepare.py train.py evaluate.py",
            "python prepare.py --output data/dataset.json --seed 7",
            "python train.py --dataset data/dataset.json --output artifacts/final_model.json",
            "python evaluate.py --dataset data/dataset.json --model artifacts/final_model.json --metrics-output artifacts/final_metrics.json --max-val-bpb 3.75",
            "pytest -q",
        ],
        "memory": True,
        "prepare_command": "python prepare.py --output data/dataset.json --seed 7",
        "train_command": "python {train_file} --dataset data/dataset.json --output {model_file}",
        "evaluate_command": "python evaluate.py --dataset data/dataset.json --model {model_file} --metrics-output {metrics_file} {incumbent_args}",
        "metric_name": "val_bpb",
        "metric_direction": "minimize",
        "tie_breaker_metric": "train_bpb",
        "stagnation_limit": 2,
        "body": "Use a compiled proposal and implementation loop to improve train.py.",
    }

    exit_code = run_compiled_task(task_payload, repo_root=repo_root)
    state = json.loads((repo_root / ".loopstack" / "state.json").read_text())

    assert exit_code == 0
    assert state["status"] == "passed"
    assert state["best"]["metrics"]["val_bpb"] <= state["baseline"]["metrics"]["val_bpb"]
    assert "order: int = 2" in (repo_root / "train.py").read_text()
