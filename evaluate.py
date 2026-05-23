"""Measure what DPO actually changed.

Two complementary metrics:

1. **Reward margin on held-out preferences** -- on the HH test split, how often
   does pi_theta assign higher log-prob to the chosen response than to the
   rejected one, relative to pi_ref? This is the metric DPO directly optimises;
   if it doesn't go up, training didn't work.

2. **GPT-judge head-to-head** (optional) -- generate from both models on a fixed
   set of prompts, ask an external judge to pick a winner. Approximates "did
   anyone actually prefer the new model?"

We always run (1). (2) is gated on having an Anthropic/OpenAI key set.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.no_grad()
def conditional_logprob(model, tokenizer, prompt: str, response: str, device) -> float:
    """Sum log-prob of `response` tokens given `prompt`, ignoring the prompt itself."""
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    full_ids = tokenizer(prompt + response, return_tensors="pt").input_ids.to(device)

    labels = full_ids.clone()
    labels[:, : prompt_ids.shape[1]] = -100  # mask prompt

    out = model(full_ids, labels=labels)
    # HF returns mean NLL over unmasked tokens; recover the sum.
    n_resp_tokens = (labels != -100).sum().item()
    if n_resp_tokens == 0:
        return 0.0
    return float(-out.loss.item() * n_resp_tokens)


def eval_preferences(base_model, dpo_model, tokenizer, dataset, device, n=200):
    """For each (prompt, chosen, rejected), compute the DPO implicit reward
    delta = beta * [log pi(y|x) - log pi_ref(y|x)] and report margin / accuracy.
    We drop the beta because we only need the sign of the difference.
    """
    margins, accs = [], []
    for i, ex in enumerate(dataset.select(range(min(n, len(dataset))))):
        lp_c_dpo = conditional_logprob(dpo_model, tokenizer, ex["prompt"], ex["chosen"], device)
        lp_r_dpo = conditional_logprob(dpo_model, tokenizer, ex["prompt"], ex["rejected"], device)
        lp_c_ref = conditional_logprob(base_model, tokenizer, ex["prompt"], ex["chosen"], device)
        lp_r_ref = conditional_logprob(base_model, tokenizer, ex["prompt"], ex["rejected"], device)

        r_chosen = lp_c_dpo - lp_c_ref
        r_rejected = lp_r_dpo - lp_r_ref
        margin = r_chosen - r_rejected
        margins.append(margin)
        accs.append(1.0 if margin > 0 else 0.0)

        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{n}] running margin mean={sum(margins)/len(margins):+.3f}  acc={sum(accs)/len(accs):.2%}")

    return {
        "n": len(margins),
        "mean_margin": sum(margins) / len(margins),
        "preference_accuracy": sum(accs) / len(accs),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--dpo", required=True, help="Path to DPO-trained model or adapter.")
    p.add_argument("--data", default="data/hh_qwen")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--out", default="results/eval.json")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float16 if device != "cpu" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model: {args.base}")
    base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=dtype).to(device).eval()

    print(f"Loading DPO model:  {args.dpo}")
    # Try to load as a full model first; if that fails it's a LoRA adapter.
    try:
        dpo = AutoModelForCausalLM.from_pretrained(args.dpo, torch_dtype=dtype).to(device).eval()
    except (OSError, ValueError):
        from peft import PeftModel
        backbone = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=dtype).to(device)
        dpo = PeftModel.from_pretrained(backbone, args.dpo).to(device).eval()

    test_ds = load_from_disk(f"{args.data}/test")

    print(f"\nEvaluating implicit rewards on {min(args.n, len(test_ds))} held-out preferences...")
    results = eval_preferences(base, dpo, tokenizer, test_ds, device, n=args.n)

    print("\n=== Results ===")
    print(json.dumps(results, indent=2))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
