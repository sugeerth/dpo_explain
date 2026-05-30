# DPO from Scratch — Qwen2.5-0.5B on HH-RLHF

End-to-end Direct Preference Optimization (DPO) pipeline. One small model, one
preference dataset, one GPU. Written to be the thing I wished existed when I
first sat down to actually *run* DPO instead of just reading the paper.

- **Model:** [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) (494M params)
- **Data:** [`Anthropic/hh-rlhf`](https://huggingface.co/datasets/Anthropic/hh-rlhf), 8k preference pairs
- **Loss:** DPO (Rafailov et al., 2023), β=0.1
- **Hardware:** Colab T4 (free), single 24 GB GPU, or Apple Silicon MPS
- **Time:** ~35 min on a T4 with LoRA

→ **Blog post:** [`BLOG.md`](BLOG.md) — *"I ran DPO. Here's what I saw."* (800 words on what the loss actually does to the model, and what surprised me.)

---

## Why this repo

It's easy to nod through the DPO paper and still not know:
- what the chosen/rejected log-probs actually do during training,
- why the reference model is in the loss (and what happens without it),
- what β trades off in practice,
- where HH-RLHF lies to you.

So I trained the smallest model that's still a real chat model, on the canonical
preference dataset, and watched the metrics. The repo is the artifact. The blog
is what I'd tell a colleague over coffee.

## The DPO loss in one line

$$
\mathcal{L}_{\text{DPO}} = - \mathbb{E}_{(x, y_w, y_l)} \Bigg[ \log \sigma \Big( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \Big) \Bigg]
$$

Push the chosen response's log-prob ratio up. Push the rejected one down. Stay
near the reference model (β controls how tightly). No reward model, no PPO,
just supervised classification on preference pairs.

## "From scratch" — and how I know it's right

The equation above is the whole repo, so I wrote it out in plain PyTorch rather
than only calling a library. [`dpo_loss.py`](dpo_loss.py) is the two functions
that matter: `batch_logps` (per-token log-probs → masked sum, with the causal
shift-by-one done explicitly) and `dpo_loss` (the `logsigmoid` of the margin).
[`train_dpo_scratch.py`](train_dpo_scratch.py) is a hand-written training loop —
explicit frozen reference model, manual batching, AdamW, grad accumulation — the
twin of the TRL-based `train_dpo.py`.

A from-scratch loss is only worth anything if it's *correct*, so I tie it out
against TRL numerically. [`test_dpo_loss.py`](test_dpo_loss.py) checks three things:

1. **Formula** — `batch_logps` matches TRL's own `selective_log_softmax` primitive.
2. **End-to-end** — the loss a real `trl.DPOTrainer` computes on a batch equals
   the loss my code computes on the same batch, **to ~3e-5**.
3. **Tokenization** — my collator produces byte-identical `input_ids` /
   `completion_mask` to TRL's pipeline.

```bash
python -m pytest test_dpo_loss.py -v   # 8 passed
python verify_against_trl.py           # [loss] TRL=16.33763885 ours=16.33767319 |diff|=3.43e-05
```

> Note on TRL versions: TRL rewrote its DPO internals after 0.11 — the old
> `DPOTrainer.get_batch_logps` static method is gone, and per-token masking now
> uses a separate `completion_mask` tensor instead of `-100` labels. The tie-out
> targets the current path (verified on `trl==0.29.0`); see `requirements.txt`.

## Repo layout

```
prepare_data.py       # Anthropic/hh-rlhf -> {prompt, chosen, rejected} with Qwen chat template
dpo_loss.py           # The DPO loss, hand-written in plain PyTorch (the "from scratch" part)
train_dpo.py          # TRL DPOTrainer wrapper, ~140 lines (the "ship it" version)
train_dpo_scratch.py  # Hand-written training loop using dpo_loss.py
train_sft.py          # SFT-on-chosen baseline (the control group)
test_dpo_loss.py      # Proves the hand-written loss == TRL, to ~1e-4
verify_against_trl.py # Same tie-out, as a standalone script you read the numbers off
evaluate.py           # Implicit-reward margin + preference accuracy (+ length controls)
generate.py           # Side-by-side base vs DPO sampling on fixed prompts
run_colab.ipynb       # One-click reproduction in Colab
BLOG.md               # The 800-word post
results/
  sample_outputs.md   # Un-cherry-picked base-vs-DPO completions
  eval.json           # Numbers from evaluate.py
```

## Quickstart

### Local (Linux/Mac, ≥16 GB VRAM or LoRA on a small GPU)

```bash
pip install -r requirements.txt

# 1. Build dataset (downloads ~80 MB from HF, takes ~2 min)
python3 prepare_data.py --n_train 8000 --n_eval 500

# 2. Train. Either the TRL version (ship it) or the from-scratch loop (read it):
python3 train_dpo.py --lora --batch_size 2 --grad_accum 8          # TRL DPOTrainer
python3 train_dpo_scratch.py --batch_size 2 --grad_accum 8         # hand-written loop

# 2b. (Optional) the control group: SFT on the chosen responses only
python3 train_sft.py --batch_size 2 --grad_accum 8

# 3. Score it (mean margin, preference accuracy, length controls)
python3 evaluate.py --dpo checkpoints/qwen05b-dpo --n 200
python3 generate.py --dpo checkpoints/qwen05b-dpo
```

### Colab (free T4)

Open [`run_colab.ipynb`](run_colab.ipynb) → Runtime → Run all. ~35 min start to
finish, no setup.

### Apple Silicon (MPS)

```bash
python3 prepare_data.py --n_train 1000 --n_eval 100
python3 train_dpo.py --lora --batch_size 1 --grad_accum 16 --max_length 768
```

Works. Slow (~2 hr for 1k examples on M2). I'd recommend Colab unless you're
allergic to the cloud.

## What "it worked" looks like

After training, `evaluate.py` should print something like:

```json
{
  "n": 200,
  "mean_margin": 0.42,
  "preference_accuracy": 0.64,
  "frac_chosen_longer": 0.71,
  "margin_lengthdiff_corr": 0.18,
  "length_matched_accuracy": 0.61,
  "n_length_matched": 48
}
```

- **`mean_margin`** is the average of `(log π_θ(y_w|x) − log π_ref(y_w|x)) − (log π_θ(y_l|x) − log π_ref(y_l|x))` over held-out HH pairs. Positive means the DPO model assigns relatively more probability to the human-preferred response than the base model does. This is the loss's own definition of "did it learn."
- **`preference_accuracy`** is the fraction of test pairs where that margin is positive. Base model is 0.50 by construction; DPO typically lands in the 0.60–0.70 range on HH at 1 epoch.
- **Length controls.** Summed log-probs are length-sensitive and HH "chosen"
  responses are often just longer (`frac_chosen_longer`), so the margin could be
  a length artifact. `margin_lengthdiff_corr` near 0 says it isn't; and
  `length_matched_accuracy` re-measures accuracy on only the pairs whose two
  responses are within 5 tokens of each other. If that stays above 0.5, the
  signal survives length. (Inspired by [Park et al., 2024](https://arxiv.org/abs/2403.19159).)

Run `train_sft.py` and score it the same way to see how much of the margin a
plain SFT-on-chosen baseline already buys you — usually a lot, which is exactly
why the baseline belongs in the repo.

If your margin is negative or your accuracy is below 0.55, β is too small, the
LR is too high, or your dataset got truncated. The blog post talks about which.

## Things this repo deliberately does *not* do

- **No reward model / no PPO.** That's the whole point of DPO.
- **No exotic tricks.** No SimPO, no IPO, no KTO, no length penalty. The
  un-tweaked DPO is the baseline anyone interested would compare against.
- **No huge dataset.** 8k examples is enough to make the loss move and the
  numbers move. Scaling is a different exercise.
- **No claim that the resulting model is good.** A 0.5B Qwen DPO'd on HH for
  one epoch is a research artifact, not a chat product. See `results/` for
  honest side-by-side outputs — some are better, some are worse, some are just
  *different*.

## References

- Rafailov et al., 2023. *Direct Preference Optimization: Your Language Model is Secretly a Reward Model.* [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)
- Bai et al., 2022. *Training a Helpful and Harmless Assistant with RLHF.* (HH-RLHF dataset.) [arXiv:2204.05862](https://arxiv.org/abs/2204.05862)
- Hugging Face TRL: <https://github.com/huggingface/trl>

## License

MIT for the code. The model and dataset retain their original licenses (Qwen
license, MIT for HH-RLHF).
