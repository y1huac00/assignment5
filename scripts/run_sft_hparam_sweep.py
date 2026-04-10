"""
Run the GSM8K SFT experiment on the full training set while sweeping
learning rate and gradient accumulation steps.

Example:

python scripts/run_sft_hparam_sweep.py \
    --learning_rates 1e-5 2e-5 5e-5 \
    --gradient_accumulation_steps 4 8 16 \
    --num_evals 10 \
    --eval_backend vllm \
    --train_gpu 0 \
    --eval_gpu 1
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_LEARNING_RATES = ["5e-6", "1e-5", "2e-5"]
DEFAULT_GRADIENT_ACCUMULATION_STEPS = ["4", "8", "16"]


def has_flag(args: list[str], flag: str) -> bool:
    return flag in args


def has_option(args: list[str], option: str) -> bool:
    return option in args


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def slugify_learning_rate(value: str) -> str:
    return value.lower().replace(".", "p").replace("-", "m")


def build_run_name(learning_rate: str, gradient_accumulation_steps: int) -> str:
    lr_slug = slugify_learning_rate(learning_rate)
    return f"lr_{lr_slug}__ga_{gradient_accumulation_steps}"


def build_run_command(
    run_out_dir: Path,
    run_name: str,
    wandb_group: str,
    learning_rate: str,
    train_batch_size: int,
    gradient_accumulation_steps: int,
    forwarded_args: list[str],
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "cs336_alignment.sft_experiment",
        "--out_dir",
        str(run_out_dir),
        "--learning_rate",
        learning_rate,
        "--train_batch_size",
        str(train_batch_size),
        "--gradient_accumulation_steps",
        str(gradient_accumulation_steps),
    ]

    cmd.extend(forwarded_args)
    if has_flag(forwarded_args, "--use_wandb"):
        if not has_option(forwarded_args, "--wandb_group"):
            cmd.extend(["--wandb_group", wandb_group])
        if not has_option(forwarded_args, "--wandb_name"):
            cmd.extend(["--wandb_name", run_name])
    return cmd


def summarize_run(
    run_out_dir: Path,
    learning_rate: str,
    train_batch_size: int,
    gradient_accumulation_steps: int,
) -> dict[str, Any]:
    summary = {
        "learning_rate": float(learning_rate),
        "learning_rate_raw": learning_rate,
        "train_batch_size": train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "out_dir": str(run_out_dir),
    }

    config_path = run_out_dir / "config.json"
    history_path = run_out_dir / "history.json"
    test_metrics_path = run_out_dir / "test_metrics.json"

    if config_path.exists():
        config = load_json(config_path)
        summary["model_name"] = config.get("model_name")
        summary["gradient_accumulation_steps"] = config.get("gradient_accumulation_steps")
        summary["effective_batch_size"] = (
            config.get("train_batch_size", 0) * config.get("gradient_accumulation_steps", 0)
        )
        summary["num_epochs"] = config.get("num_epochs")

    if history_path.exists():
        history = load_json(history_path)
        train_history = history.get("train", [])
        val_history = history.get("val", [])
        summary["num_optimizer_steps"] = len(train_history)
        summary["num_val_evals"] = len(val_history)
        if val_history:
            best_val = max(val_history, key=lambda item: item.get("reward", float("-inf")))
            summary["best_val_reward"] = best_val.get("reward")
            summary["best_val_accuracy"] = best_val.get("accuracy")
            summary["best_val_step"] = best_val.get("step")

    if test_metrics_path.exists():
        test_metrics = load_json(test_metrics_path)
        summary["test_reward"] = test_metrics.get("reward")
        summary["test_accuracy"] = test_metrics.get("accuracy")
        summary["test_num_examples"] = test_metrics.get("num_examples")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--learning_rates",
        nargs="+",
        default=DEFAULT_LEARNING_RATES,
        help="Learning rates to evaluate.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        nargs="+",
        default=DEFAULT_GRADIENT_ACCUMULATION_STEPS,
        help="Gradient accumulation steps to evaluate.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Fixed microbatch size to use for every run.",
    )
    parser.add_argument(
        "--base_out_dir",
        type=str,
        default="./outputs/gsm8k_sft_hparam_sweep",
        help="Directory under which per-run experiment folders will be created.",
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
    wandb_group = base_out_dir.name

    learning_rates = args.learning_rates
    gradient_accumulation_steps_list = [int(value) for value in args.gradient_accumulation_steps]
    results = []

    for learning_rate in learning_rates:
        for gradient_accumulation_steps in gradient_accumulation_steps_list:
            run_name = build_run_name(learning_rate, gradient_accumulation_steps)
            run_out_dir = base_out_dir / run_name
            test_metrics_path = run_out_dir / "test_metrics.json"

            if args.skip_if_complete and test_metrics_path.exists():
                print(f"[skip] {run_name} already has {test_metrics_path}")
            else:
                cmd = build_run_command(
                    run_out_dir=run_out_dir,
                    run_name=run_name,
                    wandb_group=wandb_group,
                    learning_rate=learning_rate,
                    train_batch_size=args.train_batch_size,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    forwarded_args=forwarded_args,
                )
                print(
                    f"[run] lr={learning_rate} train_batch_size={args.train_batch_size} "
                    f"grad_accum={gradient_accumulation_steps}"
                )
                print(" ".join(cmd))
                subprocess.run(cmd, check=True, cwd=repo_root)

            results.append(
                summarize_run(
                    run_out_dir=run_out_dir,
                    learning_rate=learning_rate,
                    train_batch_size=args.train_batch_size,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                )
            )

    summary_path = base_out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved summary to {summary_path}")
    for result in results:
        print(
            f"lr={result['learning_rate_raw']}\t"
            f"train_batch_size={result['train_batch_size']}\t"
            f"grad_accum={result['gradient_accumulation_steps']}\t"
            f"best_val_reward={result.get('best_val_reward')}\t"
            f"test_reward={result.get('test_reward')}"
        )


if __name__ == "__main__":
    main()
