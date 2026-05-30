"""SFT-on-chosen baseline -- the control group for the DPO experiment.

The honest question any reviewer asks about a DPO result on HH-RLHF is: "how
much of that did you get for free by just fine-tuning on the chosen responses?"
A surprising amount, usually. This script trains exactly that baseline -- plain
next-token cross-entropy on the *chosen* response (prompt tokens masked) -- on
the same data and hyperparameter scale as `train_dpo_scratch.py`, so the two are
comparable under `evaluate.py`.

    python train_sft.py --data data/hh_qwen --out checkpoints/qwen05b-sft
    python evaluate.py --base Qwen/Qwen2.5-0.5B-Instruct --dpo checkpoints/qwen05b-sft

(Yes, `evaluate.py --dpo` also works for the SFT model: the implicit-reward
metric just measures "how much more does THIS model prefer chosen over rejected
than the base does", which is exactly the comparison we want.)
"""

import argparse

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from train_dpo import pick_dtype_and_device
from train_dpo_scratch import tokenize_pair


def collate_sft(examples: list[dict], tokenizer, max_length: int | None = None) -> dict:
    """input_ids = prompt+chosen+eos; labels mask the prompt and padding with -100."""
    pad_id = tokenizer.pad_token_id
    rows = []  # (input_ids, labels)
    for ex in examples:
        prompt_ids, chosen_ids, _ = tokenize_pair(ex, tokenizer)
        ids = prompt_ids + chosen_ids
        labels = [-100] * len(prompt_ids) + list(chosen_ids)
        if max_length is not None:
            ids, labels = ids[-max_length:], labels[-max_length:]
        rows.append((ids, labels))

    width = max(len(ids) for ids, _ in rows)
    input_ids, attention_mask, label_rows = [], [], []
    for ids, labels in rows:
        pad_len = width - len(ids)
        input_ids.append(ids + [pad_id] * pad_len)
        attention_mask.append([1] * len(ids) + [0] * pad_len)
        label_rows.append(labels + [-100] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(label_rows, dtype=torch.long),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--data", default="data/hh_qwen")
    p.add_argument("--out", default="checkpoints/qwen05b-sft")
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--log_every", type=int, default=10)
    args = p.parse_args()

    torch.manual_seed(42)
    dtype, device = pick_dtype_and_device()
    print(f"Using device={device} dtype={dtype}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = load_from_disk(f"{args.data}/train")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, attn_implementation="eager")
    model.to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.train()

    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda exs: collate_sft(exs, tokenizer, max_length=args.max_length),
    )

    total_optim_steps = int((len(loader) / args.grad_accum) * args.epochs)
    if args.max_steps > 0:
        total_optim_steps = min(total_optim_steps, args.max_steps)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    scheduler = get_cosine_schedule_with_warmup(
        optim, num_warmup_steps=int(0.1 * total_optim_steps), num_training_steps=max(total_optim_steps, 1)
    )
    print(f"Examples={len(train_ds)}  optim steps={total_optim_steps}")

    step, micro = 0, 0
    optim.zero_grad()
    done = False
    for _ in range(int(args.epochs) + 1):
        if done:
            break
        for batch in loader:
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
                use_cache=False,
            )
            (out.loss / args.grad_accum).backward()
            micro += 1
            if micro % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                scheduler.step()
                optim.zero_grad()
                step += 1
                if step % args.log_every == 0:
                    print(f"step {step:4d}/{total_optim_steps}  ce_loss {out.loss.item():.4f}")
                if total_optim_steps and step >= total_optim_steps:
                    done = True
                    break

    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"\nDone. SFT baseline saved to {args.out}")
    print(f"Now: python evaluate.py --base {args.model} --dpo {args.out}")


if __name__ == "__main__":
    main()
