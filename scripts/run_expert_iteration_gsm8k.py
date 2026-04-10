"""
Run Expert Iteration on GSM8K by generating reasoning traces from the current
policy, filtering correct traces, and then SFTing on the filtered data.

Example:

python scripts/run_expert_iteration_gsm8k.py \
    --g_values 4 8 16 \
    --db_sizes 512 1024 2048 \
    --n_ei_steps 5 \
    --model_name Qwen/Qwen2.5-Math-1.5B \
    --train_gpu 0 \
    --sample_gpu 1 \
    --eval_backend vllm \
    --async_eval
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_jsonl(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def save_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_train_val(
    examples: list[dict[str, Any]],
    val_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shuffled = examples.copy()
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio))
    return shuffled[n_val:], shuffled[:n_val]


def build_prompt(question: str) -> str:
    return f"Question: {question.strip()}\nAnswer:"


def prepare_text_for_math_verify(text: str) -> str:
    text = text.strip()
    if "</answer>" in text:
        text = text[: text.index("</answer>") + len("</answer>")].strip()
    if "####" in text:
        text = text.split("####")[-1].strip()
    return text


def reward_fn(pred_text: str, gold_text: str) -> float:
    try:
        from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify

        gold_parsed = parse(
            prepare_text_for_math_verify(gold_text),
            extraction_config=(
                LatexExtractionConfig(boxed_match_priority=0),
                ExprExtractionConfig(),
            ),
            fallback_mode="no_fallback",
            extraction_mode="first_match",
            parsing_timeout=1,
        )
        pred_parsed = parse(
            prepare_text_for_math_verify(pred_text),
            extraction_config=(
                LatexExtractionConfig(boxed_match_priority=0),
                ExprExtractionConfig(),
            ),
            fallback_mode="no_fallback",
            extraction_mode="first_match",
            parsing_timeout=1,
        )
        return 1.0 if verify(gold_parsed, pred_parsed) else 0.0
    except Exception:
        return 0.0


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def has_flag(args: list[str], flag: str) -> bool:
    return flag in args


def has_option(args: list[str], option: str) -> bool:
    return option in args


def build_run_name(g_value: int, db_size: int) -> str:
    return f"G_{g_value}__Db_{db_size}"


def sample_question_batch(
    examples: list[dict[str, Any]],
    db_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    if db_size > len(examples):
        raise ValueError(f"Requested D_b={db_size}, but only {len(examples)} train examples exist.")

    rng = random.Random(seed)
    indices = rng.sample(range(len(examples)), db_size)
    return [examples[idx] for idx in indices]


def generate_reasoning_traces(
    *,
    model_path: str,
    examples: list[dict[str, Any]],
    g_value: int,
    sample_gpu: int,
    max_new_tokens: int,
    min_new_tokens: int,
    sampling_temperature: float,
    sampling_top_p: float,
    sampling_seed: int,
    vllm_gpu_memory_utilization: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    original_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(sample_gpu)
    from vllm import LLM, SamplingParams

    prompts = [build_prompt(ex["question"]) for ex in examples]
    gold_answers = [ex["answer"] for ex in examples]

    llm = None
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        trust_remote_code=True,
        gpu_memory_utilization=vllm_gpu_memory_utilization,
        enforce_eager=True,
        disable_log_stats=True,
    )
    sampling_params = SamplingParams(
        temperature=sampling_temperature,
        top_p=sampling_top_p,
        max_tokens=max_new_tokens,
        min_tokens=min_new_tokens,
        n=g_value,
        seed=sampling_seed,
    )

    all_generations: list[dict[str, Any]] = []
    filtered_examples: list[dict[str, Any]] = []

    try:
        outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
        for example, prompt, gold_answer, output in zip(examples, prompts, gold_answers, outputs):
            for candidate_idx, candidate in enumerate(output.outputs):
                prediction = candidate.text.strip()
                reward = reward_fn(prediction, gold_answer)
                generation_record = {
                    "question": example["question"],
                    "gold_answer": gold_answer,
                    "prompt": prompt,
                    "prediction": prediction,
                    "reward": reward,
                    "candidate_idx": candidate_idx,
                }
                all_generations.append(generation_record)
                if reward == 1.0:
                    filtered_examples.append(
                        {
                            "question": example["question"],
                            "answer": prediction,
                        }
                    )
    finally:
        if llm is not None:
            del llm
        gc.collect()
        if original_cuda_visible_devices is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible_devices

    return all_generations, filtered_examples


def build_sft_command(
    *,
    model_name: str,
    train_jsonl: Path,
    val_jsonl: Path,
    test_jsonl: str,
    out_dir: Path,
    run_name: str,
    wandb_group: str,
    forwarded_args: list[str],
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "cs336_alignment.sft_experiment",
        "--model_name",
        model_name,
        "--train_jsonl",
        str(train_jsonl),
        "--val_jsonl",
        str(val_jsonl),
        "--test_jsonl",
        test_jsonl,
        "--out_dir",
        str(out_dir),
    ]
    cmd.extend(forwarded_args)

    if has_flag(forwarded_args, "--use_wandb"):
        if not has_option(forwarded_args, "--wandb_group"):
            cmd.extend(["--wandb_group", wandb_group])
        if not has_option(forwarded_args, "--wandb_name"):
            cmd.extend(["--wandb_name", run_name])

    return cmd


def summarize_step(
    *,
    step_dir: Path,
    generated_count: int,
    filtered_count: int,
) -> dict[str, Any]:
    result = {
        "step_dir": str(step_dir),
        "generated_count": generated_count,
        "filtered_count": filtered_count,
        "filter_rate": (filtered_count / generated_count) if generated_count > 0 else 0.0,
    }

    history_path = step_dir / "sft_run" / "history.json"
    test_metrics_path = step_dir / "sft_run" / "test_metrics.json"

    if history_path.exists():
        history = load_json(history_path)
        val_history = history.get("val", [])
        result["num_val_evals"] = len(val_history)
        if val_history:
            best_val = max(val_history, key=lambda item: item.get("reward", float("-inf")))
            result["best_val_reward"] = best_val.get("reward")
            result["best_val_step"] = best_val.get("step")

    if test_metrics_path.exists():
        test_metrics = load_json(test_metrics_path)
        result["test_reward"] = test_metrics.get("reward")
        result["test_accuracy"] = test_metrics.get("accuracy")

    return result


def summarize_run(run_dir: Path, g_value: int, db_size: int) -> dict[str, Any]:
    summary = {
        "G": g_value,
        "D_b": db_size,
        "run_dir": str(run_dir),
        "steps": [],
    }

    run_summary_path = run_dir / "run_summary.json"
    if run_summary_path.exists():
        return load_json(run_summary_path)

    step_dirs = sorted(path for path in run_dir.iterdir() if path.is_dir() and path.name.startswith("step_"))
    for step_dir in step_dirs:
        step_summary_path = step_dir / "step_summary.json"
        if step_summary_path.exists():
            summary["steps"].append(load_json(step_summary_path))

    if summary["steps"]:
        summary["final_filtered_count"] = summary["steps"][-1].get("filtered_count")
        summary["final_best_val_reward"] = summary["steps"][-1].get("best_val_reward")
        summary["final_test_reward"] = summary["steps"][-1].get("test_reward")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonl", type=str, default="./data/gsm8k/train.jsonl")
    parser.add_argument("--test_jsonl", type=str, default="./data/gsm8k/test.jsonl")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--base_out_dir", type=str, default="./outputs/gsm8k_expert_iteration")
    parser.add_argument("--g_values", nargs="+", default=["4", "8", "16"])
    parser.add_argument("--db_sizes", nargs="+", default=["512", "1024", "2048"])
    parser.add_argument("--n_ei_steps", type=int, default=5)
    parser.add_argument("--sample_gpu", type=int, default=1)
    parser.add_argument("--sampling_temperature", type=float, default=0.7)
    parser.add_argument("--sampling_top_p", type=float, default=1.0)
    parser.add_argument("--sampling_min_tokens", type=int, default=4)
    parser.add_argument("--sampling_max_tokens", type=int, default=256)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--skip_if_complete", action="store_true")

    args, forwarded_args = parser.parse_known_args()

    repo_root = Path(__file__).resolve().parents[1]
    base_out_dir = Path(args.base_out_dir)
    base_out_dir.mkdir(parents=True, exist_ok=True)

    all_train_examples = load_jsonl(args.train_jsonl)
    train_pool, val_examples = split_train_val(
        all_train_examples,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    shared_dir = base_out_dir / "_shared"
    shared_dir.mkdir(parents=True, exist_ok=True)
    val_jsonl_path = shared_dir / "val.jsonl"
    train_pool_jsonl_path = shared_dir / "train_pool.jsonl"
    save_jsonl(str(val_jsonl_path), val_examples)
    save_jsonl(str(train_pool_jsonl_path), train_pool)

    g_values = [int(value) for value in args.g_values]
    db_sizes = [int(value) for value in args.db_sizes]
    wandb_group = base_out_dir.name
    run_summaries = []

    for g_value in g_values:
        for db_size in db_sizes:
            run_name = build_run_name(g_value, db_size)
            run_dir = base_out_dir / run_name
            run_dir.mkdir(parents=True, exist_ok=True)
            run_summary_path = run_dir / "run_summary.json"

            if args.skip_if_complete and run_summary_path.exists():
                print(f"[skip] {run_name} already has {run_summary_path}")
                run_summaries.append(load_json(run_summary_path))
                continue

            print(f"[run] {run_name}")
            current_policy_path = args.model_name
            step_summaries = []

            for step_idx in range(1, args.n_ei_steps + 1):
                step_dir = run_dir / f"step_{step_idx}"
                step_dir.mkdir(parents=True, exist_ok=True)
                step_summary_path = step_dir / "step_summary.json"

                if args.skip_if_complete and step_summary_path.exists():
                    print(f"[skip] {run_name} step={step_idx} already completed")
                    step_summary = load_json(step_summary_path)
                    step_summaries.append(step_summary)
                    next_policy_path = step_dir / "sft_run" / "best_ckpt"
                    if next_policy_path.exists():
                        current_policy_path = str(next_policy_path)
                    continue

                sampled_questions = sample_question_batch(
                    train_pool,
                    db_size=db_size,
                    seed=args.seed + (1000 * step_idx) + (100000 * g_value) + db_size,
                )

                print(
                    f"[ei] run={run_name} step={step_idx}/{args.n_ei_steps} "
                    f"sampling D_b={db_size} questions with G={g_value}"
                )
                all_generations, filtered_examples = generate_reasoning_traces(
                    model_path=str(current_policy_path),
                    examples=sampled_questions,
                    g_value=g_value,
                    sample_gpu=args.sample_gpu,
                    max_new_tokens=args.sampling_max_tokens,
                    min_new_tokens=args.sampling_min_tokens,
                    sampling_temperature=args.sampling_temperature,
                    sampling_top_p=args.sampling_top_p,
                    sampling_seed=args.seed + step_idx,
                    vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                )

                generations_jsonl = step_dir / "generated_outputs.jsonl"
                filtered_jsonl = step_dir / "filtered_sft_data.jsonl"
                save_jsonl(str(generations_jsonl), all_generations)
                save_jsonl(str(filtered_jsonl), filtered_examples)

                generated_count = len(all_generations)
                filtered_count = len(filtered_examples)
                print(
                    f"[ei] run={run_name} step={step_idx} "
                    f"generated={generated_count} filtered={filtered_count}"
                )

                if filtered_count == 0:
                    step_summary = {
                        "step": step_idx,
                        "generated_count": generated_count,
                        "filtered_count": 0,
                        "filter_rate": 0.0,
                        "skipped_sft": True,
                    }
                    with open(step_summary_path, "w", encoding="utf-8") as f:
                        json.dump(step_summary, f, ensure_ascii=False, indent=2)
                    step_summaries.append(step_summary)
                    continue

                sft_run_dir = step_dir / "sft_run"
                step_run_name = f"{run_name}__ei_step_{step_idx}"
                cmd = build_sft_command(
                    model_name=str(current_policy_path),
                    train_jsonl=filtered_jsonl,
                    val_jsonl=val_jsonl_path,
                    test_jsonl=args.test_jsonl,
                    out_dir=sft_run_dir,
                    run_name=step_run_name,
                    wandb_group=wandb_group,
                    forwarded_args=forwarded_args,
                )
                print(" ".join(cmd))
                subprocess.run(cmd, check=True, cwd=repo_root)

                step_summary = {
                    "step": step_idx,
                    **summarize_step(
                        step_dir=step_dir,
                        generated_count=generated_count,
                        filtered_count=filtered_count,
                    ),
                }
                with open(step_summary_path, "w", encoding="utf-8") as f:
                    json.dump(step_summary, f, ensure_ascii=False, indent=2)

                step_summaries.append(step_summary)
                current_policy_path = str(sft_run_dir / "best_ckpt")

            run_summary = {
                "G": g_value,
                "D_b": db_size,
                "run_dir": str(run_dir),
                "n_ei_steps": args.n_ei_steps,
                "steps": step_summaries,
            }
            if step_summaries:
                run_summary["final_filtered_count"] = step_summaries[-1].get("filtered_count")
                run_summary["final_best_val_reward"] = step_summaries[-1].get("best_val_reward")
                run_summary["final_test_reward"] = step_summaries[-1].get("test_reward")

            with open(run_summary_path, "w", encoding="utf-8") as f:
                json.dump(run_summary, f, ensure_ascii=False, indent=2)

            run_summaries.append(run_summary)

    summary_path = base_out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(run_summaries, f, ensure_ascii=False, indent=2)

    print(f"\nSaved summary to {summary_path}")
    for result in run_summaries:
        print(
            f"G={result['G']}\t"
            f"D_b={result['D_b']}\t"
            f"final_filtered={result.get('final_filtered_count')}\t"
            f"final_best_val_reward={result.get('final_best_val_reward')}\t"
            f"final_test_reward={result.get('final_test_reward')}"
        )


if __name__ == "__main__":
    main()
