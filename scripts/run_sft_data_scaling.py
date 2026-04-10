"""
Run the GSM8K SFT experiment with different training set sizes.

Example:

python scripts/run_sft_data_scaling.py \
    --model_name Qwen/Qwen2.5-Math-1.5B \
    --train_gpu 0 \
    --eval_backend torch \
    --num_epochs 1
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_TRAIN_SIZES = ["128", "256", "512", "1024", "all"]


def parse_size(size: str) -> int | None:
    if size == "all":
        return None
    return int(size)


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_run_command(
    run_out_dir: Path,
    size: str,
    forwarded_args: list[str],
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "cs336_alignment.sft_experiment",
        "--out_dir",
        str(run_out_dir),
    ]

    max_train_examples = parse_size(size)
    if max_train_examples is not None:
        cmd.extend(["--max_train_examples", str(max_train_examples)])

    cmd.extend(forwarded_args)
    return cmd


def summarize_run(run_out_dir: Path, size: str) -> dict[str, Any]:
    summary = {
        "train_size": size,
        "out_dir": str(run_out_dir),
    }

    config_path = run_out_dir / "config.json"
    history_path = run_out_dir / "history.json"
    test_metrics_path = run_out_dir / "test_metrics.json"

    if config_path.exists():
        config = load_json(config_path)
        summary["max_train_examples"] = config.get("max_train_examples")
        summary["model_name"] = config.get("model_name")

    if history_path.exists():
        history = load_json(history_path)
        train_history = history.get("train", [])
        val_history = history.get("val", [])
        summary["num_optimizer_steps"] = len(train_history)
        summary["num_val_evals"] = len(val_history)
        if val_history:
            best_val = max(val_history, key=lambda item: item.get("reward", float("-inf")))
            summary["best_val_format_reward"] = best_val.get("format_reward")
            summary["best_val_answer_reward"] = best_val.get("answer_reward")
            summary["best_val_reward"] = best_val.get("reward")
            summary["best_val_accuracy"] = best_val.get("accuracy")
            summary["best_val_step"] = best_val.get("step")

    if test_metrics_path.exists():
        test_metrics = load_json(test_metrics_path)
        summary["test_format_reward"] = test_metrics.get("format_reward")
        summary["test_answer_reward"] = test_metrics.get("answer_reward")
        summary["test_reward"] = test_metrics.get("reward")
        summary["test_accuracy"] = test_metrics.get("accuracy")
        summary["test_num_examples"] = test_metrics.get("num_examples")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=DEFAULT_TRAIN_SIZES,
        help="Training set sizes to evaluate. Use 'all' for the full training set.",
    )
    parser.add_argument(
        "--base_out_dir",
        type=str,
        default="./outputs/gsm8k_sft_data_scaling",
        help="Directory under which per-size experiment folders will be created.",
    )
    parser.add_argument(
        "--skip_if_complete",
        action="store_true",
        help="Skip a run if its output directory already contains test_metrics.json.",
    )

    args, forwarded_args = parser.parse_known_args()

    repo_root = Path(__file__).resolve().parents[1]
    base_out_dir = Path(args.base_out_dir)
    base_out_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for size in args.sizes:
        parse_size(size)
        run_out_dir = base_out_dir / f"train_{size}"
        test_metrics_path = run_out_dir / "test_metrics.json"

        if args.skip_if_complete and test_metrics_path.exists():
            print(f"[skip] train_size={size} already has {test_metrics_path}")
        else:
            cmd = build_run_command(
                run_out_dir=run_out_dir,
                size=size,
                forwarded_args=forwarded_args,
            )
            print(f"[run] train_size={size}")
            print(" ".join(cmd))
            subprocess.run(cmd, check=True, cwd=repo_root)

        results.append(summarize_run(run_out_dir, size))

    summary_path = base_out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved summary to {summary_path}")
    for result in results:
        print(
            f"train_size={result['train_size']}\t"
            f"best_val_reward={result.get('best_val_reward')}\t"
            f"best_val_answer_reward={result.get('best_val_answer_reward')}\t"
            f"test_reward={result.get('test_reward')}\t"
            f"test_answer_reward={result.get('test_answer_reward')}"
        )


if __name__ == "__main__":
    main()
