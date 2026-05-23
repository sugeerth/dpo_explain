# Base vs DPO — sample outputs

*Populated by `python3 generate.py --dpo checkpoints/qwen05b-dpo`.*

Below is the **un-cherry-picked** output from the script on a fixed set of
prompts (same prompts, same random seed for both models so differences are due
to the weights, not the noise). Examples where DPO wins, examples where it
loses, examples where the two are indistinguishable — all included.

The blog post (`../BLOG.md`) discusses what these outputs actually tell us
(spoiler: less than the implicit-reward metric does — preferences ≠ aesthetics).

> Run `generate.py` after training to fill this in.
