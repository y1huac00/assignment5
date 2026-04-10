import math
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

try:
    from .helper import get_response_log_probs, sft_microbatch_train_step
    from .sft_data import GSM8KSFTDataset, make_sft_collate_fn
except ImportError:
    from helper import get_response_log_probs, sft_microbatch_train_step
    from sft_data import GSM8KSFTDataset, make_sft_collate_fn


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
    cfg,
    device: torch.device,
    out_dir: Path,
    eval_fn: Callable,
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

    history = {"train": [], "val": []}

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

    initial_val = eval_fn(model, tokenizer, val_examples)
    initial_val["step"] = 0
    history["val"].append(initial_val)
    print(
        f"[validation] step=0 reward={initial_val['reward']:.4f} "
        f"num_examples={initial_val['num_examples']}"
    )

    best_val_reward = initial_val["reward"]
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

            loss, _ = sft_microbatch_train_step(
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
            if not should_step:
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            optimizer_step += 1
            history["train"].append(
                {
                    "step": optimizer_step,
                    "epoch": epoch,
                    "loss": running_group_loss,
                    "lr": scheduler.get_last_lr()[0],
                }
            )

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

            if optimizer_step % cfg.eval_every != 0:
                continue

            val_metrics = eval_fn(model, tokenizer, val_examples)
            val_metrics["step"] = optimizer_step
            history["val"].append(val_metrics)
            print(
                f"[validation] step={optimizer_step} "
                f"reward={val_metrics['reward']:.4f} "
                f"num_examples={val_metrics['num_examples']}"
            )

            if val_metrics["reward"] > best_val_reward:
                best_val_reward = val_metrics["reward"]
                print(f"New best validation reward: {best_val_reward:.4f} at step {optimizer_step}")
                save_checkpoint(model, tokenizer, out_dir / "best_ckpt")

    final_val = eval_fn(model, tokenizer, val_examples)
    final_val["step"] = optimizer_step
    history["val"].append(final_val)
    print(
        f"[validation] final_step={optimizer_step} "
        f"reward={final_val['reward']:.4f} "
        f"num_examples={final_val['num_examples']}"
    )

    if final_val["reward"] > best_val_reward:
        best_val_reward = final_val["reward"]
        print(f"Final model is the new best checkpoint with reward {best_val_reward:.4f}")
        save_checkpoint(model, tokenizer, out_dir / "best_ckpt")

    print("Finished SFT training.")
    return history
