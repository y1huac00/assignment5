import argparse
from dataclasses import dataclass


@dataclass
class Config:
    train_jsonl: str = "./data/gsm8k/train.jsonl"
    test_jsonl: str = "./data/gsm8k/test.jsonl"
    out_dir: str = "./outputs/gsm8k_sft"

    model_name: str = "Qwen/Qwen2.5-Math-1.5B"
    val_jsonl: str | None = None
    val_ratio: float = 0.1
    seed: int = 42

    num_epochs: int = 1
    train_batch_size: int = 4
    eval_batch_size: int = 8
    gradient_accumulation_steps: int = 8

    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0

    max_new_tokens: int = 256
    num_evals: int = 10
    log_every: int = 10
    eval_backend: str = "torch"
    async_eval: bool = False
    skip_initial_val: bool = False
    skip_test: bool = False
    use_wandb: bool = False
    wandb_project: str = "cs336-alignment"
    wandb_entity: str | None = None
    wandb_name: str | None = None
    wandb_group: str | None = None

    max_train_examples: int | None = None
    max_val_examples: int | None = 64
    max_test_examples: int | None = 128

    train_gpu: int = 0
    eval_gpu: int = 1
    vllm_gpu_memory_utilization: float = 0.85


def parse_args() -> Config:
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_jsonl", type=str, default="./data/gsm8k/train.jsonl")
    parser.add_argument("--test_jsonl", type=str, default="./data/gsm8k/test.jsonl")
    parser.add_argument("--out_dir", type=str, default="./outputs/gsm8k_sft")

    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--val_jsonl", type=str, default=None)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)

    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--num_evals", type=int, default=10)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--eval_backend", type=str, default="torch", choices=["torch", "vllm"])
    parser.add_argument("--async_eval", action="store_true")
    parser.add_argument("--skip_initial_val", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="cs336-alignment")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)

    parser.add_argument("--max_train_examples", type=int, default=None)
    parser.add_argument("--max_val_examples", type=int, default=64)
    parser.add_argument("--max_test_examples", type=int, default=128)
    parser.add_argument("--train_gpu", type=int, default=0)
    parser.add_argument("--eval_gpu", type=int, default=1)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.85)

    args = parser.parse_args()
    return Config(**vars(args))
