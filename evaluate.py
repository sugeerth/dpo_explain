"""Measure what DPO actually changed.

On the HH test split, compare pi_theta against pi_ref on held-out preferences:
how often does pi_theta assign higher log-prob to the chosen response than to
the rejected one, relative to pi_ref? This is the metric DPO directly optimises;
if it doesn't go up, training didn't work.

Because HH "chosen" responses are often just longer, the script also reports
length controls -- the chosen-longer fraction, the margin/length-diff
correlation, and a length-matched accuracy (Park et al. 2024).
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


def _response_n_tokens(tokenizer, prompt: str, response: str) -> int:
    """Token count of the response alone, matching conditional_logprob's slicing."""
    n_prompt = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
    n_full = tokenizer(prompt + response, return_tensors="pt").input_ids.shape[1]
    return max(n_full - n_prompt, 0)


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    denom = (vx * vy) ** 0.5
    return cov / denom if denom > 0 else 0.0


def eval_preferences(base_model, dpo_model, tokenizer, dataset, device, n=200):
    """For each (prompt, chosen, rejected), compute the DPO implicit reward
    delta = beta * [log pi(y|x) - log pi_ref(y|x)] and report margin / accuracy.
    We drop the beta because we only need the sign of the difference.

    We also report a length-controlled view. The HH "chosen" response is often
    just longer, and summed log-probs are length-sensitive, so a naive margin can
    be inflated by length alone. To check whether the signal survives:
      * `margin_lengthdiff_corr` -- Pearson corr between the margin and
        (len_chosen - len_rejected). Near 0 = margin is not explained by length.
      * `length_matched_accuracy` -- preference accuracy restricted to pairs
        whose response lengths are within `LEN_TOL` tokens of each other.
    """
    LEN_TOL = 5
    margins, accs, len_diffs = [], [], []
    matched_accs = []
    chosen_longer = 0

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

        len_c = _response_n_tokens(tokenizer, ex["prompt"], ex["chosen"])
        len_r = _response_n_tokens(tokenizer, ex["prompt"], ex["rejected"])
        len_diffs.append(len_c - len_r)
        chosen_longer += 1 if len_c > len_r else 0
        if abs(len_c - len_r) <= LEN_TOL:
            matched_accs.append(1.0 if margin > 0 else 0.0)

        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{n}] running margin mean={sum(margins)/len(margins):+.3f}  acc={sum(accs)/len(accs):.2%}")

    return {
        "n": len(margins),
        "mean_margin": sum(margins) / len(margins),
        "preference_accuracy": sum(accs) / len(accs),
        # Length controls
        "frac_chosen_longer": chosen_longer / len(margins),
        "margin_lengthdiff_corr": _pearson(margins, [float(d) for d in len_diffs]),
        "length_matched_accuracy": (sum(matched_accs) / len(matched_accs)) if matched_accs else None,
        "n_length_matched": len(matched_accs),
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
