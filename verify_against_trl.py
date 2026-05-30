"""Print, in one place, how closely the hand-written DPO loss matches TRL.

`test_dpo_loss.py` asserts this in CI; this script is the human-readable version
you run once and paste the numbers from into the README / blog. It builds a real
`trl.DPOTrainer`, runs one batch through it, and reports the max absolute
difference between TRL's loss / logps and ours, computed from the same batch.

    python verify_against_trl.py
    python verify_against_trl.py --model Qwen/Qwen2.5-0.5B-Instruct

Runs CPU/fp32 so the comparison is exact and device-independent.
"""

import argparse

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

import trl
from dpo_loss import dpo_loss
from train_dpo_scratch import collate_dpo, sequence_logps

EXAMPLES = [
    {"prompt": "Human: What is the capital of France?\n\nAssistant: ",
     "chosen": "The capital of France is Paris.", "rejected": "I think it is Berlin."},
    {"prompt": "Human: Recommend a book.\n\nAssistant: ",
     "chosen": "Try 'The Left Hand of Darkness' by Le Guin.", "rejected": "books are boring"},
    {"prompt": "Human: How do I boil an egg?\n\nAssistant: ",
     "chosen": "Put the egg in boiling water for about 9 minutes.", "rejected": "idk google it"},
    {"prompt": "Human: Cheer me up.\n\nAssistant: ",
     "chosen": "You've got this -- one small step at a time.", "rejected": "no."},
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--beta", type=float, default=0.1)
    args = p.parse_args()

    print(f"trl=={trl.__version__}  model={args.model}  beta={args.beta}  (CPU/fp32)\n")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).eval()
    ref = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).eval()
    # Nudge the policy off the reference so the margin (and the loss) is
    # non-trivial -- otherwise loss is exactly -log(0.5) and proves little.
    torch.manual_seed(0)
    with torch.no_grad():
        for prm in model.parameters():
            prm.add_(torch.randn_like(prm) * 0.02)

    cfg = DPOConfig(
        output_dir="/tmp/dpo_verify",
        beta=args.beta,
        loss_type="sigmoid",
        per_device_train_batch_size=len(EXAMPLES),
        max_length=512,
        bf16=False,
        fp16=False,
        use_cpu=True,
        report_to="none",
    )
    ds = Dataset.from_list(EXAMPLES)
    trainer = DPOTrainer(model=model, ref_model=ref, args=cfg, train_dataset=ds, processing_class=tok)

    rows = [trainer.train_dataset[i] for i in range(len(EXAMPLES))]
    trl_batch = trainer.data_collator(rows)

    # (1) Tokenization parity: our collate vs TRL's pipeline.
    our_batch = collate_dpo(EXAMPLES, tok)
    tok_match = all(torch.equal(our_batch[k], trl_batch[k]) for k in our_batch)
    print(f"[tokenization] our collate_dpo == TRL batch tensors : {tok_match}")

    # (2) Loss + logps tie-out on TRL's own batch.
    trainer.model.eval()
    with torch.no_grad():
        trl_loss = float(trainer._compute_loss(trainer.model, trl_batch, return_outputs=False))
        pol = sequence_logps(model, trl_batch, device="cpu")
        rf = sequence_logps(ref, trl_batch, device="cpu")
    pc, pr = pol.chunk(2, dim=0)
    rc, rr = rf.chunk(2, dim=0)
    our_loss = float(dpo_loss(pc, pr, rc, rr, beta=args.beta)[0])

    mode = "eval" if not trainer.model.training else "train"
    trl_chosen = trainer._metrics[mode]["logps/chosen"][-1]
    trl_rejected = trainer._metrics[mode]["logps/rejected"][-1]

    print(f"[loss]   TRL={trl_loss:.8f}  ours={our_loss:.8f}  |diff|={abs(trl_loss-our_loss):.2e}")
    print(f"[logp_c] TRL={trl_chosen:.6f}  ours={float(pc.mean()):.6f}  "
          f"|diff|={abs(trl_chosen-float(pc.mean())):.2e}")
    print(f"[logp_r] TRL={trl_rejected:.6f}  ours={float(pr.mean()):.6f}  "
          f"|diff|={abs(trl_rejected-float(pr.mean())):.2e}")

    ok = tok_match and abs(trl_loss - our_loss) < 1e-4
    print("\n" + ("PASS: hand-written DPO matches TRL to <1e-4." if ok else "FAIL: see diffs above."))


if __name__ == "__main__":
    main()
