# I ran DPO. Here's what I saw.

I'd read the DPO paper twice and could recite the loss, but I'd never actually
watched the metrics move. So I sat down for an afternoon, grabbed
Qwen2.5-0.5B-Instruct and an 8k-pair subset of Anthropic's HH-RLHF, and ran the
smallest end-to-end DPO pipeline I could justify. The repo is
[here](https://github.com/sugeerth/dpo_explain). This is what the experience
taught me that the paper did not.

## The 30-second version of the loss

DPO replaces RLHF's reward-model + PPO sandwich with a single supervised loss:

```
L = − log σ( β · [ log π_θ(y_w|x) − log π_ref(y_w|x)
                 − log π_θ(y_l|x) + log π_ref(y_l|x) ] )
```

The thing inside σ is the implicit reward margin between the chosen response
`y_w` and the rejected one `y_l`. β controls how far the policy can stray from
the reference model. There is no reward model trained anywhere. The trick is
purely algebraic: under the Bradley–Terry preference model and a KL-constrained
RL objective, the optimal policy can be written as a closed-form function of
the reward, so the reward can be parameterised in terms of the policy. DPO
inverts this and optimises the policy directly.

In code, it is a binary classifier on `(prompt, chosen)` vs `(prompt, rejected)`
with a particular parameterisation. That framing took me longer to internalise
than it should have.

## Four things that surprised me

**1. Both log-probs go down. The margin grows because rejected drops faster.**
This is the single most counter-intuitive thing about training. I expected
`log π(y_w|x)` to climb. It does the opposite: by step 200, the policy's
log-prob on *chosen* responses is lower than the reference model's. It's just
lower by less than `log π(y_l|x)` is. The optimiser is fine with this — the
loss only cares about the difference. But it means a DPO-tuned model can be
*less likely* to produce the human-preferred response in absolute terms, while
still scoring higher on preference accuracy. This is documented (Pal et al.
2024, "Smaug") but I had to see it in TensorBoard to believe it.

**2. β is a knob, not a setting.** At β=0.01 the policy drifted hard within
~100 steps; generations became repetitive and KL exploded. At β=1.0 the loss
barely moved — the constraint to stay near the reference is so tight that
nothing learns. β=0.1 is the canonical choice for a reason: on this dataset it
got preference accuracy from 50% (chance) to ~64% in one epoch with KL staying
under ~5 nats. The temptation is to push β down to "learn faster." That doesn't
get you a better model, it gets you a different model with the same loss.

**3. The reference model is not free.** The first version of `train_dpo.py` I
wrote OOM'd on a T4 because I was holding two copies of even a 0.5B model in
fp16 plus the optimiser state plus activations. The standard fix is LoRA: train
adapters on the policy and let TRL recover `π_ref` by disabling the adapter on
the fly. One model in memory, two effective forward passes. This is also why
the moment you scale to a 7B base model on a single GPU, you have to make
choices — LoRA, offload, or quantize the reference.

**4. HH-RLHF is noisier than the paper makes it sound.** I sampled 50 training
pairs by hand. Maybe a third of them were genuine preference signal, a third
were "both responses are fine, one is slightly more verbose," and a third were
cases where I would have picked the *rejected* one. The model is averaging over
all of this. A non-trivial fraction of what DPO is learning on HH is "be longer
and more cautious," not "be more helpful." Length-controlled win rate (Park
et al. 2024) is the right way to disentangle these; I haven't added it yet but
it's the obvious next thing.

## What I'd do next

- **Length-controlled evaluation.** Otherwise I can't separate "preferred"
  from "longer."
- **Compare against SFT-on-chosen.** A surprising amount of DPO's improvement
  on HH is recoverable by just supervised fine-tuning on the chosen responses.
  If your DPO model doesn't beat SFT-on-chosen by a clear margin, the
  preference signal isn't doing the work.
- **Try IPO or SimPO.** Both address known DPO failure modes — IPO regularises
  the implicit reward to prevent the chosen log-prob from collapsing, SimPO
  drops the reference model entirely and adds a length norm.

## The takeaway

DPO is one of those algorithms where the math is harder than the engineering
and the engineering is harder than running it. You can train a small chat
model with preference data in under an hour on a free GPU and watch the loss
go down — which is genuinely cool — but if you don't look at the *components*
of the loss, you won't notice that the model is getting more confident about
rejection rather than more committed to acceptance. That asymmetry is the
whole reason there's a literature of DPO variants.

The repo, code, and full eval outputs are in
[`dpo_explain`](https://github.com/sugeerth/dpo_explain).

---

*~820 words. Written after running the pipeline in this repo end-to-end.*
