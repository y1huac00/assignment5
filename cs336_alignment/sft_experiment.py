import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .sft_config import parse_args
    from .sft_data import (
        choose_torch_dtype,
        load_jsonl,
        maybe_truncate,
        resolve_train_device,
        save_jsonl,
        set_seed,
        split_train_val,
    )
    from .sft_eval import (
        evaluate_gsm8k,
        evaluate_gsm8k_vllm_subprocess,
        run_vllm_eval_worker,
    )
    from .sft_train import save_checkpoint, train_sft
except ImportError:
    from sft_config import parse_args
    from sft_data import (
        choose_torch_dtype,
        load_jsonl,
        maybe_truncate,
        resolve_train_device,
        save_jsonl,
        set_seed,
        split_train_val,
    )
    from sft_eval import (
        evaluate_gsm8k,
        evaluate_gsm8k_vllm_subprocess,
        run_vllm_eval_worker,
    )
    from sft_train import save_checkpoint, train_sft


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
        if cfg.eval_backend == "vllm" and (cfg.eval_gpu < 0 or cfg.eval_gpu >= gpu_count):
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

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **model_kwargs)
    model.to(device)

    if cfg.eval_backend == "vllm":

        def eval_fn(eval_model, eval_tokenizer, examples):
            with tempfile.TemporaryDirectory(prefix="vllm_eval_ckpt_") as tmpdir:
                eval_ckpt_dir = Path(tmpdir)
                save_checkpoint(eval_model, eval_tokenizer, eval_ckpt_dir)
                return evaluate_gsm8k_vllm_subprocess(
                    model_path=eval_ckpt_dir,
                    tokenizer_path=cfg.model_name,
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
            tokenizer_path=cfg.model_name,
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

        best_model = AutoModelForCausalLM.from_pretrained(best_ckpt_dir, **best_model_kwargs)
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

    print(f"Final test reward: {test_metrics['reward']:.4f}")
    print("Done.")


if __name__ == "__main__":
    if os.environ.get("CS336_VLLM_EVAL_WORKER") == "1":
        run_vllm_eval_worker()
    else:
        main()
