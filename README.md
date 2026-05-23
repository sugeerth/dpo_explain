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

## Repo layout

```
prepare_data.py   # Anthropic/hh-rlhf -> {prompt, chosen, rejected} with Qwen chat template
train_dpo.py      # TRL DPOTrainer wrapper, ~140 lines
evaluate.py       # Implicit-reward margin + preference accuracy on held-out HH
generate.py       # Side-by-side base vs DPO sampling on fixed prompts
run_colab.ipynb   # One-click reproduction in Colab
BLOG.md           # The 800-word post
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

# 2. Train (LoRA flag recommended unless you have a 24 GB GPU)
python3 train_dpo.py --lora --batch_size 2 --grad_accum 8

# 3. Score it
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
  "preference_accuracy": 0.64
}
```

- **`mean_margin`** is the average of `(log π_θ(y_w|x) − log π_ref(y_w|x)) − (log π_θ(y_l|x) − log π_ref(y_l|x))` over held-out HH pairs. Positive means the DPO model assigns relatively more probability to the human-preferred response than the base model does. This is the loss's own definition of "did it learn."
- **`preference_accuracy`** is the fraction of test pairs where that margin is positive. Base model is 0.50 by construction; DPO typically lands in the 0.60–0.70 range on HH at 1 epoch.

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
