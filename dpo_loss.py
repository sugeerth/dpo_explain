"""The DPO loss, hand-written in plain PyTorch.

This is the part the rest of the repo is named after. `train_dpo.py` uses
TRL's `DPOTrainer`, which is the sensible thing to do in practice -- but it
hides the one equation that matters. Here it is, unhidden, in two small
functions, with the exact tensor bookkeeping spelled out.

DPO (Rafailov et al., 2023) minimises

    L = - E[ log sigma( beta * ( (log pi(y_w|x) - log pi_ref(y_w|x))
                               - (log pi(y_l|x) - log pi_ref(y_l|x)) ) ) ]

So all we ever need from a model is one number per (prompt, response): the
summed log-probability of the response tokens under that model. `batch_logps`
produces that number; `dpo_loss` combines the four of them (policy/ref x
chosen/rejected) into the scalar above.

These two functions are validated to match TRL 0.29's internal computation to
~1e-4 -- see `test_dpo_loss.py` / `verify_against_trl.py`. The tensor
operations below are written to mirror `DPOTrainer._compute_loss`
(trl/trainer/dpo_trainer.py, lines ~1121-1209 in 0.29.0): the same shift-by-one,
the same completion-mask convention, the same sigmoid/logsigmoid form.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def batch_logps(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    average_log_prob: bool = False,
) -> torch.Tensor:
    """Summed (or averaged) log-prob of the masked tokens, one number per row.

    Args:
        logits:    `[B, T, V]` raw, *unshifted* model logits.
        labels:    `[B, T]` token ids, *unshifted* (i.e. the model's `input_ids`).
        loss_mask: `[B, T]` 1 for completion tokens to score, 0 for prompt/pad.
        average_log_prob: divide by the number of scored tokens (length-norm).
            Standard sigmoid DPO uses the *sum* (`False`); IPO / length-norm
            variants average. Leaving this `True` silently rescales the loss and
            thus the effective `beta`, so it defaults to `False`.

    Returns:
        `[B]` per-sequence log-probabilities.

    Why the shifts. A causal LM at position `t` predicts token `t+1`. So we
    score `logits[:, :-1]` (positions 0..T-2) against `labels[:, 1:]`
    (tokens 1..T-1). The mask is shifted the *same* way -- it indexes label
    tokens, not logit positions. Forgetting to shift the mask is the classic
    off-by-one that leaks the last prompt token into the score.
    """
    if logits.shape[:-1] != labels.shape:
        raise ValueError(
            f"logits[..., :-1] shape {tuple(logits.shape[:-1])} must match "
            f"labels shape {tuple(labels.shape)}"
        )
    if labels.shape != loss_mask.shape:
        raise ValueError("labels and loss_mask must have the same shape")

    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    shift_mask = loss_mask[:, 1:].to(shift_logits.dtype if shift_logits.is_floating_point() else torch.float32)

    # log p(token) = log_softmax(logits)[token]. Compute in fp32: bf16/fp16
    # log_softmax diverges from fp32 by >1e-3, which is the difference between
    # "matches TRL to 1e-4" and "doesn't".
    logps = torch.gather(
        shift_logits.float().log_softmax(dim=-1),
        dim=-1,
        index=shift_labels.unsqueeze(-1),
    ).squeeze(-1)  # [B, T-1]

    logps = logps * shift_mask  # zero out prompt / padding; identical to TRL's per_token_logps[mask==0]=0

    summed = logps.sum(dim=-1)
    if average_log_prob:
        return summed / shift_mask.sum(dim=-1).clamp(min=1)
    return summed


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The DPO loss itself, plus the implicit rewards for logging.

    Args (all `[B]`, summed log-probs from `batch_logps`):
        policy_chosen_logps:   log pi_theta(y_w | x)
        policy_rejected_logps: log pi_theta(y_l | x)
        ref_chosen_logps:      log pi_ref(y_w | x)
        ref_rejected_logps:    log pi_ref(y_l | x)
        beta: KL-strength. Larger keeps the policy nearer the reference.

    Returns:
        loss:             scalar, mean over the batch.
        chosen_rewards:   `[B]` beta * (log pi - log pi_ref) for chosen (detached).
        rejected_rewards: `[B]` same for rejected (detached).

    The two reward tensors are exactly TRL's `rewards/chosen` and
    `rewards/rejected`; `(chosen_rewards > rejected_rewards)` is `rewards/accuracies`.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps

    # This regroups to (chosen_logratio - rejected_logratio), TRL's `delta_score`
    # under f_divergence_type="reverse_kl" (the standard DPO setting).
    logits = pi_logratios - ref_logratios

    # F.logsigmoid, NOT log(sigmoid(.)): for a large positive margin sigmoid
    # saturates to 1.0 and log(1.0)=0 loses the gradient; logsigmoid stays exact.
    loss = -F.logsigmoid(beta * logits).mean()

    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps).detach()
    return loss, chosen_rewards, rejected_rewards
