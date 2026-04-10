import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch
from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify
from tqdm import tqdm

try:
    from .sft_data import build_prompt
except ImportError:
    from sft_data import build_prompt


def truncate_response_for_reward(text: str) -> str:
    end_tag = "</answer>"
    if end_tag in text:
        end_idx = text.index(end_tag) + len(end_tag)
        return text[:end_idx].strip()
    return text.strip()


def prepare_text_for_math_verify(text: str) -> str:
    text = truncate_response_for_reward(text).strip()
    if "####" in text:
        text = text.split("####")[-1].strip()
    return text


def reward_fn(pred_text: str, gold_text: str) -> float:
    try:
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


def summarize_rewards(rewards: list[float]) -> dict[str, float]:
    num_examples = len(rewards)
    if num_examples == 0:
        return {"num_examples": 0, "reward": 0.0, "accuracy": 0.0}

    reward = sum(rewards) / num_examples
    return {"num_examples": num_examples, "reward": reward, "accuracy": reward}


@torch.no_grad()
def generate_responses(
    model,
    tokenizer,
    prompt_strs: list[str],
    max_new_tokens: int,
    device: torch.device,
) -> list[str]:
    was_training = model.training
    model.eval()

    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    batch = tokenizer(
        prompt_strs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        add_special_tokens=False,
    )
    batch = {k: v.to(device) for k, v in batch.items()}

    input_len = batch["input_ids"].shape[1]
    generated_ids = model.generate(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    new_token_ids = generated_ids[:, input_len:]
    response_strs = tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)

    tokenizer.padding_side = old_padding_side
    if was_training:
        model.train()

    return [x.strip() for x in response_strs]


@torch.no_grad()
def evaluate_gsm8k(
    model,
    tokenizer,
    examples: list[dict[str, Any]],
    eval_batch_size: int,
    max_new_tokens: int,
    device: torch.device,
    max_logged_examples: int = 5,
) -> dict[str, Any]:
    sample_records = []
    rewards = []
    num_batches = math.ceil(len(examples) / eval_batch_size) if eval_batch_size > 0 else 0

    for start in tqdm(
        range(0, len(examples), eval_batch_size),
        total=num_batches,
        desc="Evaluating",
        leave=True,
        dynamic_ncols=True,
    ):
        batch_examples = examples[start:start + eval_batch_size]
        prompt_strs = [build_prompt(ex["question"]) for ex in batch_examples]
        gold_answers = [ex["answer"] for ex in batch_examples]

        pred_responses = generate_responses(
            model=model,
            tokenizer=tokenizer,
            prompt_strs=prompt_strs,
            max_new_tokens=max_new_tokens,
            device=device,
        )

        for prompt, pred, gold in zip(prompt_strs, pred_responses, gold_answers):
            pred = truncate_response_for_reward(pred)
            reward = reward_fn(pred, gold)
            rewards.append(reward)

            if len(sample_records) < max_logged_examples:
                sample_records.append(
                    {
                        "prompt": prompt,
                        "prediction": pred,
                        "gold": gold,
                        "reward": reward,
                    }
                )

    return {**summarize_rewards(rewards), "sample_records": sample_records}


def run_vllm_eval_worker() -> None:
    model_path = os.environ["CS336_VLLM_MODEL_PATH"]
    tokenizer_path = os.environ["CS336_VLLM_TOKENIZER_PATH"]
    payload_path = os.environ["CS336_VLLM_PAYLOAD_PATH"]
    output_path = os.environ["CS336_VLLM_OUTPUT_PATH"]

    from vllm import LLM, SamplingParams

    with open(payload_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    examples = payload["examples"]
    eval_batch_size = payload["eval_batch_size"]
    max_new_tokens = payload["max_new_tokens"]
    max_logged_examples = payload["max_logged_examples"]
    gpu_memory_utilization = payload["gpu_memory_utilization"]

    llm = LLM(
        model=model_path,
        tokenizer=tokenizer_path,
        tensor_parallel_size=1,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=True,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_new_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    sample_records = []
    rewards = []
    num_batches = math.ceil(len(examples) / eval_batch_size) if eval_batch_size > 0 else 0

    for start in tqdm(
        range(0, len(examples), eval_batch_size),
        total=num_batches,
        desc="Evaluating (vLLM)",
        leave=True,
        dynamic_ncols=True,
    ):
        batch_examples = examples[start:start + eval_batch_size]
        prompt_strs = [build_prompt(ex["question"]) for ex in batch_examples]
        gold_answers = [ex["answer"] for ex in batch_examples]

        outputs = llm.generate(prompt_strs, sampling_params, use_tqdm=False)
        pred_responses = [truncate_response_for_reward(out.outputs[0].text) for out in outputs]

        for prompt, pred, gold in zip(prompt_strs, pred_responses, gold_answers):
            reward = reward_fn(pred, gold)
            rewards.append(reward)

            if len(sample_records) < max_logged_examples:
                sample_records.append(
                    {
                        "prompt": prompt,
                        "prediction": pred,
                        "gold": gold,
                        "reward": reward,
                    }
                )

    metrics = {**summarize_rewards(rewards), "sample_records": sample_records}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False)


def evaluate_gsm8k_vllm_subprocess(
    model_path: Path,
    tokenizer_path: str,
    examples: list[dict[str, Any]],
    eval_batch_size: int,
    max_new_tokens: int,
    eval_gpu: int,
    vllm_gpu_memory_utilization: float,
    max_logged_examples: int = 5,
) -> dict[str, Any]:
    payload = {
        "examples": examples,
        "eval_batch_size": eval_batch_size,
        "max_new_tokens": max_new_tokens,
        "max_logged_examples": max_logged_examples,
        "gpu_memory_utilization": vllm_gpu_memory_utilization,
    }

    with tempfile.TemporaryDirectory(prefix="vllm_eval_") as tmpdir:
        payload_path = Path(tmpdir) / "payload.json"
        output_path = Path(tmpdir) / "output.json"
        with open(payload_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(eval_gpu)
        env["CS336_VLLM_EVAL_WORKER"] = "1"
        env["CS336_VLLM_MODEL_PATH"] = str(model_path)
        env["CS336_VLLM_TOKENIZER_PATH"] = tokenizer_path
        env["CS336_VLLM_PAYLOAD_PATH"] = str(payload_path)
        env["CS336_VLLM_OUTPUT_PATH"] = str(output_path)

        cmd = [sys.executable, "-m", "cs336_alignment.sft_experiment"]
        proc = subprocess.run(cmd, env=env, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"vLLM eval subprocess failed. Return code: {proc.returncode}")

        with open(output_path, "r", encoding="utf-8") as f:
            return json.load(f)
