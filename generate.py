"""Side-by-side base vs DPO generations on a fixed prompt set.

Useful for eyeballing whether the model is now more helpful/harmless or whether
it just got shorter / more cautious / weirder. Cherry-pick from the output for
the blog post; keep the un-cherry-picked file in results/ for honesty.
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_PROMPTS = [
    "I want to learn how to invest my first $1000. Where do I start?",
    "My friend just told me they're feeling really down. What should I say?",
    "Explain why airplanes can fly to a curious 10-year-old.",
    "I'm a vegetarian visiting Tokyo for a week. What are 5 things I should eat?",
    "Write a haiku about debugging a memory leak at 2am.",
    "Should I tell my manager that I'm interviewing elsewhere? Pros and cons.",
    "I keep procrastinating on my thesis. What's actually wrong and how do I fix it?",
    "What's a good way to teach a beginner Python without overwhelming them?",
]


def load_model(name_or_path, base_name, device, dtype):
    try:
        model = AutoModelForCausalLM.from_pretrained(name_or_path, torch_dtype=dtype).to(device)
    except (OSError, ValueError):
        from peft import PeftModel
        backbone = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=dtype).to(device)
        model = PeftModel.from_pretrained(backbone, name_or_path).to(device)
    model.eval()
    return model


@torch.no_grad()
def generate(model, tokenizer, prompt: str, device, max_new_tokens=256) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )
    response = tokenizer.decode(out[0, inputs.input_ids.shape[1] :], skip_special_tokens=True)
    return response.strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--dpo", required=True)
    p.add_argument("--out", default="results/sample_outputs.md")
    p.add_argument("--prompts_file", default=None, help="Optional: path to a text file with one prompt per line.")
    args = p.parse_args()

    if args.prompts_file:
        prompts = [l.strip() for l in Path(args.prompts_file).read_text().splitlines() if l.strip()]
    else:
        prompts = DEFAULT_PROMPTS

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float16 if device != "cpu" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base...")
    base = load_model(args.base, args.base, device, dtype)
    print("Loading DPO...")
    dpo = load_model(args.dpo, args.base, device, dtype)

    torch.manual_seed(0)
    lines = ["# Base vs DPO -- side-by-side\n",
             f"Base: `{args.base}` | DPO: `{args.dpo}`\n",
             "Sampling: temp=0.7 top_p=0.9, seed=0. No cherry-picking below.\n"]

    for i, prompt in enumerate(prompts, 1):
        print(f"[{i}/{len(prompts)}] {prompt[:60]}...")
        torch.manual_seed(i)
        base_out = generate(base, tokenizer, prompt, device)
        torch.manual_seed(i)  # same seed -> same noise; differences are real
        dpo_out = generate(dpo, tokenizer, prompt, device)

        lines.append(f"\n---\n\n### Prompt {i}\n\n> {prompt}\n")
        lines.append(f"\n**Base**\n\n{base_out}\n")
        lines.append(f"\n**DPO**\n\n{dpo_out}\n")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
