"""DPO fine-tune Qwen2.5-0.5B with a hand-written training loop.

This is `train_dpo.py`'s twin. Same model, same data, same hyperparameters --
but instead of `trl.DPOTrainer` doing the work, the loss lives in `dpo_loss.py`
and the loop, the reference model, and the batching are all spelled out here.

The point is not to beat TRL (TRL is what you'd ship). The point is that every
number on the screen can be traced to a line you can read. The loss this loop
computes matches TRL's to ~1e-4 -- see `verify_against_trl.py`.

Defaults match `train_dpo.py` so the resulting model is directly comparable:
  beta=0.1, lr=5e-6, 1 epoch, batch 2, grad-accum 8, max_length 1024.

Reference model: full-rank and frozen, held explicitly in memory. (TRL's LoRA
path recovers pi_ref by toggling adapters off; that's a neat memory trick but it
hides the reference, which is the opposite of the goal here.) Full-rank 0.5B
policy + frozen ref + grad-checkpointing fits a 16 GB T4 at batch 2.

Run:
    python train_dpo_scratch.py --data data/hh_qwen --out checkpoints/qwen05b-dpo-scratch
"""

import argparse

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from dpo_loss import batch_logps, dpo_loss
# Reuse the hardware/dtype logic and model construction from the TRL script so
# the two trainers never drift on device handling.
from train_dpo import build_models, pick_dtype_and_device


def _with_eos(text: str, eos: str) -> str:
    return text if text.endswith(eos) else text + eos


def tokenize_pair(example: dict, tokenizer) -> tuple[list[int], list[int], list[int]]:
    """Return (prompt_ids, chosen_ids, rejected_ids), mirroring TRL's tokenize_fn.

    The two rules that matter (and that TRL also follows):
      1. EOS is appended to each response *as text* before tokenizing, so the
         model learns where to stop.
      2. prompt and prompt+response are tokenized *jointly*, then the prompt
         prefix is sliced off. Tokenizing the response on its own and
         concatenating ids can merge tokens across the boundary and shift
         everything by one -- a silent, miserable bug.
    """
    eos = tokenizer.eos_token
    chosen = _with_eos(example["chosen"], eos)
    rejected = _with_eos(example["rejected"], eos)

    prompt_ids = tokenizer(example["prompt"])["input_ids"]
    pc_ids = tokenizer(example["prompt"] + chosen)["input_ids"]
    pr_ids = tokenizer(example["prompt"] + rejected)["input_ids"]

    return prompt_ids, pc_ids[len(prompt_ids):], pr_ids[len(prompt_ids):]


def collate_dpo(examples: list[dict], tokenizer, max_length: int | None = None) -> dict:
    """Tokenize and pad a batch into TRL's `[chosen-rows ; rejected-rows]` layout.

    Returns a dict with `input_ids`, `attention_mask`, `completion_mask`, each
    `[2B, T]`. The first B rows are prompt+chosen, the next B are prompt+rejected,
    so `tensor.chunk(2, dim=0)` recovers the two halves. `completion_mask` is 1
    on response tokens, 0 on prompt and padding.
    """
    pad_id = tokenizer.pad_token_id
    chosen_rows, rejected_rows = [], []  # each entry: (input_ids, completion_mask)

    for ex in examples:
        prompt_ids, chosen_ids, rejected_ids = tokenize_pair(ex, tokenizer)
        pc = prompt_ids + chosen_ids
        pr = prompt_ids + rejected_ids
        pc_mask = [0] * len(prompt_ids) + [1] * len(chosen_ids)
        pr_mask = [0] * len(prompt_ids) + [1] * len(rejected_ids)

        if max_length is not None:
            # Left-truncate the prompt so the completion (and recent context)
            # always survives; never drop response tokens we score against.
            pc, pc_mask = pc[-max_length:], pc_mask[-max_length:]
            pr, pr_mask = pr[-max_length:], pr_mask[-max_length:]

        chosen_rows.append((pc, pc_mask))
        rejected_rows.append((pr, pr_mask))

    rows = chosen_rows + rejected_rows  # chosen block first, then rejected block
    width = max(len(ids) for ids, _ in rows)

    input_ids, attention_mask, completion_mask = [], [], []
    for ids, mask in rows:
        pad_len = width - len(ids)
        input_ids.append(ids + [pad_id] * pad_len)
        attention_mask.append([1] * len(ids) + [0] * pad_len)
        completion_mask.append(mask + [0] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "completion_mask": torch.tensor(completion_mask, dtype=torch.long),
    }


def sequence_logps(model, batch: dict, device) -> torch.Tensor:
    """Forward `batch` through `model` and return per-row summed completion logps."""
    out = model(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        use_cache=False,
    )
    return batch_logps(out.logits, batch["input_ids"].to(device), batch["completion_mask"].to(device))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--data", default="data/hh_qwen")
    p.add_argument("--out", default="checkpoints/qwen05b-dpo-scratch")
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--max_steps", type=int, default=-1, help="Cap steps for a quick smoke run.")
    p.add_argument("--log_every", type=int, default=10)
    args = p.parse_args()

    torch.manual_seed(42)
    dtype, device = pick_dtype_and_device()
    print(f"Using device={device} dtype={dtype}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = load_from_disk(f"{args.data}/train")

    policy, ref_model = build_models(args.model, dtype, use_lora=False)
    policy.to(device)
    ref_model.to(device)
    policy.config.use_cache = False
    policy.gradient_checkpointing_enable()
    policy.train()

    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda exs: collate_dpo(exs, tokenizer, max_length=args.max_length),
    )

    micro_per_epoch = len(loader)
    total_optim_steps = int((micro_per_epoch / args.grad_accum) * args.epochs)
    if args.max_steps > 0:
        total_optim_steps = min(total_optim_steps, args.max_steps)

    optim = torch.optim.AdamW(
        [pp for pp in policy.parameters() if pp.requires_grad], lr=args.lr, weight_decay=0.0
    )
    scheduler = get_cosine_schedule_with_warmup(
        optim,
        num_warmup_steps=int(0.1 * total_optim_steps),
        num_training_steps=max(total_optim_steps, 1),
    )

    print(f"Examples={len(train_ds)}  micro-batches/epoch={micro_per_epoch}  optim steps={total_optim_steps}")

    step, micro = 0, 0
    optim.zero_grad()
    done = False
    for epoch in range(int(args.epochs) + 1):
        if done:
            break
        for batch in loader:
            policy_logps = sequence_logps(policy, batch, device)
            with torch.no_grad():
                ref_logps = sequence_logps(ref_model, batch, device)

            pol_chosen, pol_rejected = policy_logps.chunk(2, dim=0)
            ref_chosen, ref_rejected = ref_logps.chunk(2, dim=0)
            loss, chosen_rewards, rejected_rewards = dpo_loss(
                pol_chosen, pol_rejected, ref_chosen, ref_rejected, beta=args.beta
            )

            (loss / args.grad_accum).backward()
            micro += 1

            if micro % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optim.step()
                scheduler.step()
                optim.zero_grad()
                step += 1

                if step % args.log_every == 0:
                    acc = (chosen_rewards > rejected_rewards).float().mean().item()
                    margin = (chosen_rewards - rejected_rewards).mean().item()
                    print(
                        f"step {step:4d}/{total_optim_steps}  loss {loss.item():.4f}  "
                        f"rewards/chosen {chosen_rewards.mean().item():+.3f}  "
                        f"rewards/rejected {rejected_rewards.mean().item():+.3f}  "
                        f"margin {margin:+.3f}  acc {acc:.2%}"
                    )

                if total_optim_steps and step >= total_optim_steps:
                    done = True
                    break

    policy.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"\nDone. Model saved to {args.out}")
    print(f"Now: python evaluate.py --base {args.model} --dpo {args.out}")


if __name__ == "__main__":
    main()
