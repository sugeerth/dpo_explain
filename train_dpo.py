"""DPO fine-tune Qwen2.5-0.5B-Instruct on a HH-RLHF subset.

Defaults are tuned to run on:
  * Colab T4 (16 GB) with LoRA in ~30-40 min for 8k examples
  * A single 24 GB consumer GPU full-rank
  * Apple Silicon (MPS) with --lora and a small dataset (slow but works)

The DPO loss optimises:

    L = - E[ log sigmoid( beta * ( log pi(y_w|x)/pi_ref(y_w|x)
                                 - log pi(y_l|x)/pi_ref(y_l|x) ) ) ]

so the only things you really need to log are the two log-prob ratios and the
margin between them. TRL surfaces these as `rewards/chosen`, `rewards/rejected`,
`rewards/margins`, `rewards/accuracies`.
"""

import argparse

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


def pick_dtype_and_device() -> tuple[torch.dtype, str]:
    if torch.cuda.is_available():
        major = torch.cuda.get_device_capability(0)[0]
        # bf16 needs Ampere (sm_80) or newer; T4 is sm_75 -> fp16.
        return (torch.bfloat16 if major >= 8 else torch.float16), "cuda"
    if torch.backends.mps.is_available():
        return torch.float16, "mps"
    return torch.float32, "cpu"


def build_models(model_name: str, dtype: torch.dtype, use_lora: bool):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, attn_implementation="eager"
    )
    if use_lora:
        # When the policy is a LoRA adapter we don't need a separate frozen
        # ref model -- TRL disables the adapter on the fly to score under pi_ref.
        from peft import LoraConfig, get_peft_model

        lora_cfg = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
        ref_model = None
    else:
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, attn_implementation="eager"
        )
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)
    return model, ref_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--data", default="data/hh_qwen")
    p.add_argument("--out", default="checkpoints/qwen05b-dpo")
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--max_prompt_length", type=int, default=512)
    p.add_argument("--lora", action="store_true", help="Train with LoRA (saves VRAM, skips ref model).")
    p.add_argument("--report_to", default="none", choices=["none", "wandb"])
    args = p.parse_args()

    dtype, device = pick_dtype_and_device()
    print(f"Using device={device} dtype={dtype}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = load_from_disk(f"{args.data}/train")
    eval_ds = load_from_disk(f"{args.data}/test")

    model, ref_model = build_models(args.model, dtype, use_lora=args.lora)

    cfg = DPOConfig(
        output_dir=args.out,
        beta=args.beta,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=400,
        save_total_limit=2,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16 and device == "cuda"),
        remove_unused_columns=False,
        report_to=args.report_to,
        run_name="qwen05b-dpo-hh",
        seed=42,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds.select(range(min(256, len(eval_ds)))),
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)

    print(f"\nDone. Adapter/model saved to {args.out}")
    print("Now: python evaluate.py --base", args.model, "--dpo", args.out)


if __name__ == "__main__":
    main()
