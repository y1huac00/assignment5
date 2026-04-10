import math
import shutil
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


def promote_checkpoint(src_dir: Path, dst_dir: Path) -> None:
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir)
    print(f"Promoted checkpoint to best: {dst_dir}")


def train_sft(
    model,
    tokenizer,
    train_examples: list[dict[str, Any]],
    val_examples: list[dict[str, Any]],
    cfg,
    device: torch.device,
    out_dir: Path,
    eval_fn: Callable,
    async_eval_manager=None,
    wandb_run=None,
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
    num_evals = max(1, cfg.num_evals)
    scheduled_eval_steps = sorted(
        {
            min(total_optimizer_steps, max(1, math.ceil(i * total_optimizer_steps / num_evals)))
            for i in range(1, num_evals + 1)
        }
    )
    print("Starting SFT training...")
    print(f"Train examples: {len(train_examples)}")
    print(f"Val examples:   {len(val_examples)}")
    print(f"Epochs:         {cfg.num_epochs}")
    print(f"Train batch:    {cfg.train_batch_size}")
    print(f"Grad accum:     {cfg.gradient_accumulation_steps}")
    print(f"Effective batch:{effective_batch_size}")
    print(
        "Scheduled evals:"
        f" {len(scheduled_eval_steps)} time(s) at optimizer steps {scheduled_eval_steps}"
    )
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
    if wandb_run is not None:
        wandb_run.log(
            {
                "val/step": 0,
                "val/reward": initial_val["reward"],
                "val/accuracy": initial_val["accuracy"],
                "val/num_examples": initial_val["num_examples"],
            }
        )

    best_val_reward = initial_val["reward"]
    save_checkpoint(model, tokenizer, out_dir / "best_ckpt")

    def handle_val_result(step: int, val_metrics: dict[str, Any], ckpt_dir: Path | None = None) -> None:
        nonlocal best_val_reward

        val_metrics["step"] = step
        history["val"].append(val_metrics)
        print(
            f"[validation] step={step} "
            f"reward={val_metrics['reward']:.4f} "
            f"num_examples={val_metrics['num_examples']}"
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "val/step": step,
                    "val/reward": val_metrics["reward"],
                    "val/accuracy": val_metrics["accuracy"],
                    "val/num_examples": val_metrics["num_examples"],
                    "val/best_reward_so_far": max(best_val_reward, val_metrics["reward"]),
                }
            )

        if val_metrics["reward"] > best_val_reward:
            best_val_reward = val_metrics["reward"]
            print(f"New best validation reward: {best_val_reward:.4f} at step {step}")
            if ckpt_dir is None:
                save_checkpoint(model, tokenizer, out_dir / "best_ckpt")
            else:
                promote_checkpoint(ckpt_dir, out_dir / "best_ckpt")

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
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/step": optimizer_step,
                        "train/epoch": epoch + 1,
                        "train/loss": running_group_loss,
                        "train/lr": scheduler.get_last_lr()[0],
                    }
                )

            running_group_loss = 0.0
            microbatches_in_group = 0

            if async_eval_manager is not None:
                for completed in async_eval_manager.poll():
                    handle_val_result(
                        step=completed.step,
                        val_metrics=completed.metrics,
                        ckpt_dir=completed.ckpt_dir,
                    )
                    async_eval_manager.cleanup_checkpoint(completed.ckpt_dir)

            if optimizer_step not in scheduled_eval_steps:
                continue

            if async_eval_manager is not None:
                async_eval_manager.submit(optimizer_step, model, tokenizer)
                continue

            val_metrics = eval_fn(model, tokenizer, val_examples)
            handle_val_result(step=optimizer_step, val_metrics=val_metrics)

    if async_eval_manager is not None:
        for completed in async_eval_manager.poll():
            handle_val_result(
                step=completed.step,
                val_metrics=completed.metrics,
                ckpt_dir=completed.ckpt_dir,
            )
            async_eval_manager.cleanup_checkpoint(completed.ckpt_dir)

        if async_eval_manager.last_submitted_step != optimizer_step:
            async_eval_manager.submit(optimizer_step, model, tokenizer)

        print("[async-eval] waiting for outstanding validation jobs...")
        final_val = None
        for completed in async_eval_manager.wait_for_all():
            handle_val_result(
                step=completed.step,
                val_metrics=completed.metrics,
                ckpt_dir=completed.ckpt_dir,
            )
            if completed.step == optimizer_step:
                final_val = completed.metrics
            async_eval_manager.cleanup_checkpoint(completed.ckpt_dir)

        if final_val is not None:
            print(
                f"[validation] final_step={optimizer_step} "
                f"reward={final_val['reward']:.4f} "
                f"num_examples={final_val['num_examples']}"
            )
    else:
        if optimizer_step in scheduled_eval_steps:
            final_val = history["val"][-1]
            print(
                f"[validation] final_step={optimizer_step} "
                f"reward={final_val['reward']:.4f} "
                f"num_examples={final_val['num_examples']}"
            )
        else:
            final_val = eval_fn(model, tokenizer, val_examples)
            handle_val_result(step=optimizer_step, val_metrics=final_val)
            print(
                f"[validation] final_step={optimizer_step} "
                f"reward={final_val['reward']:.4f} "
                f"num_examples={final_val['num_examples']}"
            )

    print("Finished SFT training.")
    if wandb_run is not None:
        wandb_run.summary["best_val_reward"] = best_val_reward
    return history
