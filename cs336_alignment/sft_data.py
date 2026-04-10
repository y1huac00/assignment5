import json
import random
from typing import Any

import torch
from torch.utils.data import Dataset

try:
    from .helper import tokenize_prompt_and_output
except ImportError:
    from helper import tokenize_prompt_and_output


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
    return " " + answer.strip()


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
