"""Prove the hand-written DPO loss matches TRL.

This file is the reason the repo can call itself "DPO from scratch" honestly:
the loss in `dpo_loss.py` is independent code, and these tests show it computes
the same numbers TRL does, to ~1e-4. Three levels, increasingly integrated:

  Level A  -- formula tie-out. Given identical logits/labels/mask, our
              `batch_logps` equals TRL's own `selective_log_softmax` primitive,
              and our `dpo_loss` equals the inlined Rafailov formula. No model,
              no network; runs anywhere, can't drift across TRL minor versions.

  Level B  -- end-to-end tie-out. Build a real `trl.DPOTrainer` on a tiny model,
              let it tokenize/collate/forward, and check the loss IT computes
              equals the loss WE compute from the same batch, to 1e-4.

  Level C  -- tokenization parity. Our `collate_dpo` produces the same
              input_ids/attention_mask/completion_mask as TRL's pipeline.

Levels B and C need a small model + TRL's 0.29-style internals, so they're
gated on version and skipped (not failed) if the model can't be fetched offline.

Run:  python -m pytest test_dpo_loss.py -v
"""

import pytest
import torch

from dpo_loss import batch_logps, dpo_loss

# Prefer the tiny CI fixture; fall back to the real model this project uses,
# which is cached locally even when the network is sandboxed.
CANDIDATE_MODELS = [
    "trl-internal-testing/tiny-Qwen2ForCausalLM",
    "Qwen/Qwen2.5-0.5B-Instruct",
]

try:
    import trl

    TRL_VERSION = tuple(int(x) for x in trl.__version__.split(".")[:2])
except Exception:  # pragma: no cover - trl always present here, but be safe
    trl = None
    TRL_VERSION = (0, 0)


# --------------------------------------------------------------------------- #
# Level A: formula tie-out (always runs, no model, no network)
# --------------------------------------------------------------------------- #

def test_batch_logps_matches_naive_log_softmax_gather():
    """Our summed logp == naive log_softmax->gather->mask->sum, with the right shift."""
    torch.manual_seed(0)
    B, T, V = 4, 9, 32
    logits = torch.randn(B, T, V)
    labels = torch.randint(0, V, (B, T))
    mask = torch.zeros(B, T, dtype=torch.long)
    mask[:, 4:] = 1  # first 4 positions are "prompt"

    ours = batch_logps(logits, labels, mask)

    # Independent reference: shift by one, gather, mask the *shifted* mask, sum.
    lp = logits[:, :-1, :].float().log_softmax(-1)
    gathered = lp.gather(-1, labels[:, 1:].unsqueeze(-1)).squeeze(-1)
    expected = (gathered * mask[:, 1:]).sum(-1)

    assert torch.allclose(ours, expected, atol=1e-5)


@pytest.mark.skipif(trl is None, reason="trl not importable")
def test_batch_logps_matches_trl_selective_log_softmax():
    """Tie out against TRL's *actual* primitive, the one that survived 0.29."""
    from trl.trainer.utils import selective_log_softmax

    torch.manual_seed(1)
    B, T, V = 3, 11, 40
    logits = torch.randn(B, T, V)
    labels = torch.randint(0, V, (B, T))
    mask = torch.zeros(B, T, dtype=torch.long)
    mask[:, 5:] = 1

    ours = batch_logps(logits, labels, mask)

    # Exactly TRL's _compute_loss lines: shift, selective_log_softmax, mask, sum.
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_mask = mask[:, 1:].contiguous()
    ptl = selective_log_softmax(shift_logits, shift_labels)
    ptl[shift_mask == 0] = 0.0
    expected = ptl.sum(dim=1)

    assert torch.allclose(ours, expected, atol=1e-4)


def test_batch_logps_average_log_prob():
    torch.manual_seed(2)
    B, T, V = 2, 8, 16
    logits = torch.randn(B, T, V)
    labels = torch.randint(0, V, (B, T))
    mask = torch.zeros(B, T, dtype=torch.long)
    mask[:, 3:] = 1

    summed = batch_logps(logits, labels, mask, average_log_prob=False)
    averaged = batch_logps(logits, labels, mask, average_log_prob=True)
    n_tokens = mask[:, 1:].sum(-1)
    assert torch.allclose(averaged, summed / n_tokens, atol=1e-5)


def test_dpo_loss_matches_inlined_formula():
    torch.manual_seed(3)
    pc = torch.randn(5)
    pr = torch.randn(5)
    rc = torch.randn(5)
    rr = torch.randn(5)
    beta = 0.1

    loss, ch_rew, rj_rew = dpo_loss(pc, pr, rc, rr, beta=beta)

    import torch.nn.functional as F

    delta = (pc - pr) - (rc - rr)
    expected_loss = -F.logsigmoid(beta * delta).mean()
    assert torch.allclose(loss, expected_loss, atol=1e-6)
    assert torch.allclose(ch_rew, beta * (pc - rc), atol=1e-6)
    assert torch.allclose(rj_rew, beta * (pr - rr), atol=1e-6)


def test_dpo_loss_shapes_and_zero_margin():
    """Identical policy and reference -> margin 0 -> loss = -log(0.5)."""
    z = torch.zeros(4)
    loss, ch, rj = dpo_loss(z, z, z, z, beta=0.1)
    assert torch.allclose(loss, torch.tensor(0.6931), atol=1e-3)  # -log(0.5)
    assert ch.shape == (4,) and rj.shape == (4,)


# --------------------------------------------------------------------------- #
# Level B + C: end-to-end tie-out against a real DPOTrainer
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def tiny_setup():
    """Load a tiny Qwen2 model + tokenizer, or skip if offline."""
    if TRL_VERSION < (0, 20):
        pytest.skip(f"Level B/C target TRL 0.29 internals; have {trl.__version__ if trl else 'none'}")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    errors = []
    for name in CANDIDATE_MODELS:
        try:
            tok = AutoTokenizer.from_pretrained(name)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32).eval()
            ref = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32).eval()
            # Perturb the policy so it differs from the reference; otherwise the
            # margin is 0 and the loss is trivially -log(0.5), which wouldn't
            # exercise the log-ratio computation we're trying to verify.
            torch.manual_seed(0)
            with torch.no_grad():
                for prm in model.parameters():
                    prm.add_(torch.randn_like(prm) * 0.02)
            return model, ref, tok
        except Exception as e:  # offline / cache miss -> try the next candidate
            errors.append(f"{name}: {type(e).__name__}")
    pytest.skip(f"No candidate model loadable offline ({'; '.join(errors)})")


@pytest.fixture(scope="module")
def examples():
    return [
        {"prompt": "Human: What is 2+2?\n\nAssistant: ", "chosen": "It is 4.", "rejected": "I don't know."},
        {"prompt": "Human: Name a color.\n\nAssistant: ", "chosen": "Blue is a color.", "rejected": "Banana."},
        {"prompt": "Human: Say hi.\n\nAssistant: ", "chosen": "Hello there!", "rejected": "..."},
    ]


def _build_trainer(model, ref, tok, examples, tmp_path):
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    ds = Dataset.from_list(examples)
    cfg = DPOConfig(
        output_dir=str(tmp_path),
        beta=0.1,
        loss_type="sigmoid",
        per_device_train_batch_size=len(examples),
        max_length=512,
        bf16=False,
        fp16=False,
        use_cpu=True,  # keep everything on CPU/fp32 so the tie-out is exact and device-matched
        report_to="none",
    )
    trainer = DPOTrainer(model=model, ref_model=ref, args=cfg, train_dataset=ds, processing_class=tok)
    return trainer


def test_level_b_loss_ties_out_to_trl(tiny_setup, examples, tmp_path):
    """The loss the real DPOTrainer computes == the loss we compute, to 1e-4."""
    from train_dpo_scratch import sequence_logps

    model, ref, tok = tiny_setup
    trainer = _build_trainer(model, ref, tok, examples, tmp_path)

    # Pull one batch through TRL's own tokenizer + collator.
    rows = [trainer.train_dataset[i] for i in range(len(examples))]
    batch = trainer.data_collator(rows)

    # --- TRL's loss ---
    trainer.model.eval()
    with torch.no_grad():
        trl_loss = trainer._compute_loss(trainer.model, batch, return_outputs=False)
    trl_loss = float(trl_loss)

    # --- Our loss, from the same batch tensors ---
    with torch.no_grad():
        pol = sequence_logps(model, batch, device="cpu")
        rf = sequence_logps(ref, batch, device="cpu")
    pc, pr = pol.chunk(2, dim=0)
    rc, rr = rf.chunk(2, dim=0)
    our_loss, _, _ = dpo_loss(pc, pr, rc, rr, beta=0.1)

    assert abs(float(our_loss) - trl_loss) < 1e-4, f"ours={float(our_loss)} trl={trl_loss}"


def test_level_b_logps_tie_out_to_trl_metrics(tiny_setup, examples, tmp_path):
    """Our mean chosen/rejected logps == the ones TRL logs."""
    from train_dpo_scratch import sequence_logps

    model, ref, tok = tiny_setup
    trainer = _build_trainer(model, ref, tok, examples, tmp_path)
    rows = [trainer.train_dataset[i] for i in range(len(examples))]
    batch = trainer.data_collator(rows)

    trainer.model.eval()
    with torch.no_grad():
        trainer._compute_loss(trainer.model, batch, return_outputs=False)
        pol = sequence_logps(model, batch, device="cpu")
    pc, pr = pol.chunk(2, dim=0)

    mode = "eval" if not trainer.model.training else "train"
    trl_chosen = trainer._metrics[mode]["logps/chosen"][-1]
    trl_rejected = trainer._metrics[mode]["logps/rejected"][-1]

    # These are *summed* logps (magnitudes of tens), and TRL reduces them with a
    # per-row logsumexp loop while we use log_softmax -- so they agree only to
    # float summation order, ~1e-3, not the 1e-5 the per-token path hits (Level A).
    assert abs(float(pc.mean()) - trl_chosen) < 2e-3
    assert abs(float(pr.mean()) - trl_rejected) < 2e-3


def test_level_c_tokenization_parity(tiny_setup, examples, tmp_path):
    """Our collate_dpo produces byte-identical batch tensors to TRL's pipeline."""
    from train_dpo_scratch import collate_dpo

    model, ref, tok = tiny_setup
    trainer = _build_trainer(model, ref, tok, examples, tmp_path)
    rows = [trainer.train_dataset[i] for i in range(len(examples))]
    trl_batch = trainer.data_collator(rows)

    our_batch = collate_dpo(examples, tok)  # no truncation; examples are short

    for key in ("input_ids", "attention_mask", "completion_mask"):
        assert torch.equal(our_batch[key], trl_batch[key]), f"{key} mismatch"
