import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Dataset 
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from helper import (
    tokenize_prompt_and_output,
    get_response_log_probs,
    sft_microbatch_train_step,
)

@dataclass
class Config:
    train_jsonl: str = './data/gsm8k/train.jsonl'
    test_jsonl: str = './data/gsm8k/test.jsonl'
    out_dir: str = './outputs/gsm8k_sft'

    model_name: str = "Qwen/Qwen2.5-Math-1.5B"
    val_jsonl: str | None = None
    val_ratio: float = 0.1
    seed: int = 42

    num_epochs: int = 1
    train_batch_size: int = 8
    eval_batch_size: int = 16
    gradient_accumulation_steps: int = 2

    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0

    max_new_tokens: int = 256
    eval_every: int = 100
    log_every: int = 10
    eval_backend: str = "torch"

    max_train_examples: int | None = None
    max_val_examples: int | None = None
    max_test_examples: int | None = None

    train_gpu: int = 0
    eval_gpu: int = 1
    vllm_gpu_memory_utilization: float = 0.85

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_jsonl(path: str) -> list[dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def save_jsonl(path: str, data: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for ex in data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

def split_train_val(
    examples: list[dict[str, Any]],
    val_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    examples = examples.copy()
    rng = random.Random(seed)
    rng.shuffle(examples)

    n_val = max(1, int(len(examples) * val_ratio))
    val_examples = examples[:n_val]
    train_examples = examples[n_val:]
    return train_examples, val_examples

def maybe_truncate(
    examples: list[dict[str, Any]],
    max_examples: int | None,
) -> list[dict[str, Any]]:
    if max_examples is None:
        return examples
    return examples[:max_examples]

def build_prompt(question: str) -> str:
    question = question.strip()
    return f"Question: {question}\nAnswer:"


def build_train_response(answer: str) -> str:
    # 前面补一个空格，通常对 tokenizer 更自然一些
    return " " + answer.strip()


def extract_final_answer(text: str) -> str:
    """
    GSM8K 常见格式是:
    '... reasoning ... #### 72'
    我们优先取 #### 后面的最终答案。
    如果没有 ####，就取最后一个数字。
    """
    text = text.replace("<answer>", " ").replace("</answer>", " ").strip()

    if "####" in text:
        text = text.split("####")[-1].strip()

    text = text.replace(",", "")
    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    if matches:
        return matches[-1]

    return " ".join(text.lower().split())


def is_correct_gsm8k(pred_text: str, gold_text: str) -> bool:
    return extract_final_answer(pred_text) == extract_final_answer(gold_text)


def choose_torch_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def resolve_train_device(train_gpu: int) -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")

    gpu_count = torch.cuda.device_count()
    if train_gpu < 0 or train_gpu >= gpu_count:
        raise ValueError(
            f"Requested train_gpu={train_gpu}, but only {gpu_count} CUDA device(s) are available."
        )

    return torch.device(f"cuda:{train_gpu}")


class GSM8KSFTDataset(Dataset):
    def __init__(self, examples: list[dict[str, Any]]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.examples[idx]


def make_sft_collate_fn(tokenizer):
    def collate_fn(examples: list[dict[str, Any]]) -> dict[str, Any]:
        prompt_strs = [build_prompt(ex["question"]) for ex in examples]
        output_strs = [build_train_response(ex["answer"]) for ex in examples]

        tokenized = tokenize_prompt_and_output(
            prompts=prompt_strs,
            outputs=output_strs,
            tokenizer=tokenizer,
        )

        tokenized["prompt_strs"] = prompt_strs
        tokenized["output_strs"] = output_strs
        return tokenized

    return collate_fn

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
        do_sample=False,   # 验证时先用 greedy
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    new_token_ids = generated_ids[:, input_len:]
    response_strs = tokenizer.batch_decode(
        new_token_ids,
        skip_special_tokens=True,
    )

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
    all_correct = 0
    total = 0
    sample_records = []

    for start in range(0, len(examples), eval_batch_size):
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
            correct = is_correct_gsm8k(pred, gold)
            all_correct += int(correct)
            total += 1

            if len(sample_records) < max_logged_examples:
                sample_records.append({
                    "prompt": prompt,
                    "prediction": pred,
                    "gold": gold,
                    "pred_final": extract_final_answer(pred),
                    "gold_final": extract_final_answer(gold),
                    "correct": correct,
                })

    accuracy = all_correct / total if total > 0 else 0.0

    return {
        "accuracy": accuracy,
        "num_examples": total,
        "sample_records": sample_records,
    }


def run_vllm_eval_worker() -> None:
    model_path = os.environ["CS336_VLLM_MODEL_PATH"]
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
        tensor_parallel_size=1,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_new_tokens,
    )

    all_correct = 0
    total = 0
    sample_records = []

    for start in range(0, len(examples), eval_batch_size):
        batch_examples = examples[start:start + eval_batch_size]
        prompt_strs = [build_prompt(ex["question"]) for ex in batch_examples]
        gold_answers = [ex["answer"] for ex in batch_examples]

        outputs = llm.generate(prompt_strs, sampling_params)
        pred_responses = [out.outputs[0].text.strip() for out in outputs]

        for prompt, pred, gold in zip(prompt_strs, pred_responses, gold_answers):
            correct = is_correct_gsm8k(pred, gold)
            all_correct += int(correct)
            total += 1

            if len(sample_records) < max_logged_examples:
                sample_records.append(
                    {
                        "prompt": prompt,
                        "prediction": pred,
                        "gold": gold,
                        "pred_final": extract_final_answer(pred),
                        "gold_final": extract_final_answer(gold),
                        "correct": correct,
                    }
                )

    accuracy = all_correct / total if total > 0 else 0.0
    metrics = {
        "accuracy": accuracy,
        "num_examples": total,
        "sample_records": sample_records,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False)


def evaluate_gsm8k_vllm_subprocess(
    model_path: Path,
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
        env["CS336_VLLM_PAYLOAD_PATH"] = str(payload_path)
        env["CS336_VLLM_OUTPUT_PATH"] = str(output_path)

        cmd = [sys.executable, str(Path(__file__).resolve())]
        proc = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "vLLM eval subprocess failed.\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            )

        with open(output_path, "r", encoding="utf-8") as f:
            return json.load(f)

def save_checkpoint(model, tokenizer, save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Saved checkpoint to: {save_dir}")

def train_sft(
    model,
    tokenizer,
    train_examples: list[dict[str, Any]],
    val_examples: list[dict[str, Any]],
    cfg: Config,
    device: torch.device,
    out_dir: Path,
    eval_fn,
) -> dict[str, Any]:
    train_dataset = GSM8KSFTDataset(train_examples)
    collate_fn = make_sft_collate_fn(tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=False,
    )

    if len(train_loader) == 0:
        raise ValueError("train_loader is empty. Check your train set size and batch size.")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    updates_per_epoch = math.ceil(len(train_loader) / cfg.gradient_accumulation_steps)
    total_optimizer_steps = updates_per_epoch * cfg.num_epochs
    warmup_steps = int(total_optimizer_steps * cfg.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimizer_steps,
    )

    history = {
        "train": [],
        "val": [],
    }

    effective_batch_size = cfg.train_batch_size * cfg.gradient_accumulation_steps
    print("Starting SFT training...")
    print(f"Train examples: {len(train_examples)}")
    print(f"Val examples:   {len(val_examples)}")
    print(f"Epochs:         {cfg.num_epochs}")
    print(f"Train batch:    {cfg.train_batch_size}")
    print(f"Grad accum:     {cfg.gradient_accumulation_steps}")
    print(f"Effective batch:{effective_batch_size}")
    print(f"Eval every:     {cfg.eval_every} optimizer step(s)")
    print(f"Log every:      {cfg.log_every} optimizer step(s)")
    print(f"Learning rate:  {cfg.learning_rate}")
    print(f"Device:         {device}")

    # 先做一次训练前验证，方便画 curve
    initial_val = eval_fn(model, tokenizer, val_examples)
    initial_val["step"] = 0
    history["val"].append(initial_val)
    print(
        f"[validation] step=0 accuracy={initial_val['accuracy']:.4f} "
        f"num_examples={initial_val['num_examples']}"
    )

    best_val_acc = initial_val["accuracy"]
    save_checkpoint(model, tokenizer, out_dir / "best_ckpt")

    optimizer.zero_grad(set_to_none=True)

    optimizer_step = 0
    running_group_loss = 0.0
    microbatches_in_group = 0

    for epoch in range(cfg.num_epochs):
        model.train()
        print(f"\nEpoch {epoch + 1}/{cfg.num_epochs}")

        remainder = len(train_loader) % cfg.gradient_accumulation_steps

        progress_bar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch + 1}/{cfg.num_epochs}",
        )
        for micro_idx, batch in progress_bar:
            if remainder != 0 and micro_idx >= len(train_loader) - remainder:
                current_accum_steps = remainder
            else:
                current_accum_steps = cfg.gradient_accumulation_steps

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            response_mask = batch["response_mask"].to(device)

            score_out = get_response_log_probs(
                model=model,
                input_ids=input_ids,
                labels=labels,
                return_token_entropy=False,
            )
            policy_log_probs = score_out["log_probs"]

            normalize_constant = max(float(response_mask.sum().item()), 1.0)

            loss, metadata = sft_microbatch_train_step(
                policy_log_probs=policy_log_probs,
                response_mask=response_mask,
                gradient_accumulation_steps=current_accum_steps,
                normalize_constant=normalize_constant,
            )

            running_group_loss += float(loss.detach().item())
            microbatches_in_group += 1

            should_step = (
                microbatches_in_group == current_accum_steps
                or micro_idx == len(train_loader) - 1
            )

            if should_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                optimizer_step += 1

                history["train"].append({
                    "step": optimizer_step,
                    "epoch": epoch,
                    "loss": running_group_loss,
                    "lr": scheduler.get_last_lr()[0],
                })

                progress_bar.set_postfix(
                    step=optimizer_step,
                    loss=f"{running_group_loss:.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                )
                if optimizer_step == 1 or optimizer_step % cfg.log_every == 0:
                    print(
                        f"[train] epoch={epoch + 1} step={optimizer_step} "
                        f"loss={running_group_loss:.4f} "
                        f"lr={scheduler.get_last_lr()[0]:.2e}"
                    )

                running_group_loss = 0.0
                microbatches_in_group = 0

                if optimizer_step % cfg.eval_every == 0:
                    val_metrics = eval_fn(model, tokenizer, val_examples)
                    val_metrics["step"] = optimizer_step
                    history["val"].append(val_metrics)
                    print(
                        f"[validation] step={optimizer_step} "
                        f"accuracy={val_metrics['accuracy']:.4f} "
                        f"num_examples={val_metrics['num_examples']}"
                    )

                    if val_metrics["accuracy"] > best_val_acc:
                        best_val_acc = val_metrics["accuracy"]
                        print(
                            f"New best validation accuracy: {best_val_acc:.4f} "
                            f"at step {optimizer_step}"
                        )
                        save_checkpoint(model, tokenizer, out_dir / "best_ckpt")

    # 训练结束后再做一次验证
    final_val = eval_fn(model, tokenizer, val_examples)
    final_val["step"] = optimizer_step
    history["val"].append(final_val)
    print(
        f"[validation] final_step={optimizer_step} "
        f"accuracy={final_val['accuracy']:.4f} "
        f"num_examples={final_val['num_examples']}"
    )

    if final_val["accuracy"] > best_val_acc:
        best_val_acc = final_val["accuracy"]
        print(f"Final model is the new best checkpoint with accuracy {best_val_acc:.4f}")
        save_checkpoint(model, tokenizer, out_dir / "best_ckpt")

    save_checkpoint(model, tokenizer, out_dir / "last_ckpt")
    print("Finished SFT training.")

    return history

def parse_args() -> Config:
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_jsonl", type=str, default='./data/gsm8k/train.jsonl')
    parser.add_argument("--test_jsonl", type=str, default='./data/gsm8k/test.jsonl')
    parser.add_argument("--out_dir", type=str, default='./outputs/gsm8k_sft')

    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--val_jsonl", type=str, default=None)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)

    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--eval_backend", type=str, default="torch", choices=["torch", "vllm"])

    parser.add_argument("--max_train_examples", type=int, default=None)
    parser.add_argument("--max_val_examples", type=int, default=None)
    parser.add_argument("--max_test_examples", type=int, default=None)
    parser.add_argument("--train_gpu", type=int, default=0)
    parser.add_argument("--eval_gpu", type=int, default=1)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.85)

    args = parser.parse_args()
    return Config(**vars(args))


def main():
    cfg = parse_args()
    set_seed(cfg.seed)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

    raw_train_examples = load_jsonl(cfg.train_jsonl)
    test_examples = load_jsonl(cfg.test_jsonl)

    if cfg.val_jsonl is not None and Path(cfg.val_jsonl).exists():
        train_examples = raw_train_examples
        val_examples = load_jsonl(cfg.val_jsonl)
    else:
        train_examples, val_examples = split_train_val(
            raw_train_examples,
            val_ratio=cfg.val_ratio,
            seed=cfg.seed,
        )
        save_jsonl(str(out_dir / "train_split.jsonl"), train_examples)
        save_jsonl(str(out_dir / "val_split.jsonl"), val_examples)

    train_examples = maybe_truncate(train_examples, cfg.max_train_examples)
    val_examples = maybe_truncate(val_examples, cfg.max_val_examples)
    test_examples = maybe_truncate(test_examples, cfg.max_test_examples)

    print(f"train size: {len(train_examples)}")
    print(f"val size:   {len(val_examples)}")
    print(f"test size:  {len(test_examples)}")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = choose_torch_dtype()
    device = resolve_train_device(cfg.train_gpu)

    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        if cfg.eval_backend == "vllm":
            if cfg.eval_gpu < 0 or cfg.eval_gpu >= gpu_count:
                raise ValueError(
                    f"Requested eval_gpu={cfg.eval_gpu}, but only {gpu_count} CUDA device(s) are available."
                )

        print(f"Using train GPU: cuda:{cfg.train_gpu}")
        if cfg.eval_backend == "vllm":
            print(f"Using eval GPU for vLLM subprocess: cuda:{cfg.eval_gpu}")
    else:
        if cfg.eval_backend == "vllm":
            raise ValueError("eval_backend=vllm requires CUDA, but CUDA is not available.")
        print("CUDA not available; running on CPU.")

    model_kwargs = {"trust_remote_code": True}
    if device.type == "cuda":
        model_kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        **model_kwargs,
    )
    model.to(device)

    if cfg.eval_backend == "vllm":
        eval_ckpt_dir = out_dir / "eval_ckpt"

        def eval_fn(eval_model, eval_tokenizer, examples):
            save_checkpoint(eval_model, eval_tokenizer, eval_ckpt_dir)
            return evaluate_gsm8k_vllm_subprocess(
                model_path=eval_ckpt_dir,
                examples=examples,
                eval_batch_size=cfg.eval_batch_size,
                max_new_tokens=cfg.max_new_tokens,
                eval_gpu=cfg.eval_gpu,
                vllm_gpu_memory_utilization=cfg.vllm_gpu_memory_utilization,
            )
    else:

        def eval_fn(eval_model, eval_tokenizer, examples):
            return evaluate_gsm8k(
                model=eval_model,
                tokenizer=eval_tokenizer,
                examples=examples,
                eval_batch_size=cfg.eval_batch_size,
                max_new_tokens=cfg.max_new_tokens,
                device=device,
            )

    history = train_sft(
        model=model,
        tokenizer=tokenizer,
        train_examples=train_examples,
        val_examples=val_examples,
        cfg=cfg,
        device=device,
        out_dir=out_dir,
        eval_fn=eval_fn,
    )

    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print("Reloading best checkpoint for final test evaluation...")
    best_ckpt_dir = out_dir / "best_ckpt"

    if cfg.eval_backend == "vllm":
        test_metrics = evaluate_gsm8k_vllm_subprocess(
            model_path=best_ckpt_dir,
            examples=test_examples,
            eval_batch_size=cfg.eval_batch_size,
            max_new_tokens=cfg.max_new_tokens,
            eval_gpu=cfg.eval_gpu,
            vllm_gpu_memory_utilization=cfg.vllm_gpu_memory_utilization,
        )
    else:
        best_model_kwargs = {"trust_remote_code": True}
        if device.type == "cuda":
            best_model_kwargs["torch_dtype"] = dtype

        best_model = AutoModelForCausalLM.from_pretrained(
            best_ckpt_dir,
            **best_model_kwargs,
        )
        best_model.to(device)

        test_metrics = evaluate_gsm8k(
            model=best_model,
            tokenizer=tokenizer,
            examples=test_examples,
            eval_batch_size=cfg.eval_batch_size,
            max_new_tokens=cfg.max_new_tokens,
            device=device,
        )

    with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=2)

    print(f"Final test accuracy: {test_metrics['accuracy']:.4f}")
    print("Done.")


if __name__ == "__main__":
    if os.environ.get("CS336_VLLM_EVAL_WORKER") == "1":
        run_vllm_eval_worker()
    else:
        main()
