"""Format Anthropic/hh-rlhf into the {prompt, chosen, rejected} schema TRL expects.

HH stores each example as two full dialogues that diverge only at the final
assistant turn. We split off that final turn so `prompt` is the shared context
and `chosen`/`rejected` are just the two candidate responses.
"""

import argparse
import re
from pathlib import Path

from datasets import load_dataset


HH_TURN_RE = re.compile(r"\n\n(Human|Assistant): ")


def split_prompt_response(text: str) -> tuple[str, str]:
    idx = text.rfind("\n\nAssistant: ")
    if idx == -1:
        raise ValueError("No Assistant turn found in HH example.")
    prompt = text[: idx + len("\n\nAssistant: ")]
    response = text[idx + len("\n\nAssistant: ") :]
    return prompt, response


def hh_to_messages(prompt_text: str) -> list[dict]:
    """Convert HH 'Human:/Assistant:' prompt context into chat messages."""
    body = prompt_text.strip()
    # Drop the trailing '\n\nAssistant:' marker so we don't emit an empty turn.
    body = re.sub(r"\n\nAssistant:\s*$", "", body)

    parts = HH_TURN_RE.split("\n\n" + body)
    # parts looks like: ['', 'Human', '...text...', 'Assistant', '...', 'Human', '...']
    messages: list[dict] = []
    for role, content in zip(parts[1::2], parts[2::2]):
        role_norm = "user" if role == "Human" else "assistant"
        messages.append({"role": role_norm, "content": content.strip()})
    return messages


def format_for_dpo(example: dict, tokenizer) -> dict:
    prompt_text, chosen_resp = split_prompt_response(example["chosen"])
    _, rejected_resp = split_prompt_response(example["rejected"])

    messages = hh_to_messages(prompt_text)
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return {
        "prompt": prompt,
        "chosen": chosen_resp.strip(),
        "rejected": rejected_resp.strip(),
    }


def build(model_name: str, n_train: int, n_eval: int, out_dir: str) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    raw = load_dataset("Anthropic/hh-rlhf")
    train = raw["train"].shuffle(seed=0).select(range(n_train))
    test = raw["test"].shuffle(seed=0).select(range(n_eval))

    train = train.map(lambda ex: format_for_dpo(ex, tokenizer), remove_columns=raw["train"].column_names)
    test = test.map(lambda ex: format_for_dpo(ex, tokenizer), remove_columns=raw["test"].column_names)

    # Drop pathologically long examples so we don't waste a batch slot truncating.
    def keep(example):
        return (
            len(example["prompt"]) < 4000
            and len(example["chosen"]) < 2000
            and len(example["rejected"]) < 2000
            and len(example["chosen"]) > 5
            and len(example["rejected"]) > 5
        )

    train = train.filter(keep)
    test = test.filter(keep)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train.save_to_disk(str(out / "train"))
    test.save_to_disk(str(out / "test"))

    print(f"Saved {len(train)} train / {len(test)} test examples to {out}")
    print("\nFirst training example:")
    print("PROMPT:", train[0]["prompt"][:500], "...")
    print("CHOSEN:", train[0]["chosen"][:300])
    print("REJECTED:", train[0]["rejected"][:300])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--n_train", type=int, default=8000)
    p.add_argument("--n_eval", type=int, default=500)
    p.add_argument("--out", default="data/hh_qwen")
    args = p.parse_args()
    build(args.model, args.n_train, args.n_eval, args.out)
