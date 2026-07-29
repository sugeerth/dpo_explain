"""Unit test for the pure `_with_eos` helper in train_dpo_scratch.py.

`_with_eos` enforces an idempotent EOS suffix on each response before
tokenizing -- appending it twice would teach the model a spurious double-stop,
so the idempotency property is the thing worth pinning. The module imports
torch/transformers/trl transitively, hence importorskip.
"""

import pytest

tds = pytest.importorskip(
    "train_dpo_scratch",
    reason="train_dpo_scratch.py imports torch/transformers/trl at module scope",
)
_with_eos = tds._with_eos


def test_with_eos_appends_when_missing():
    assert _with_eos("hello", "<eos>") == "hello<eos>"


def test_with_eos_is_idempotent_when_already_present():
    assert _with_eos("hello<eos>", "<eos>") == "hello<eos>"


def test_with_eos_only_checks_the_suffix():
    # An EOS in the middle does not count as already-terminated.
    assert _with_eos("a<eos>b", "<eos>") == "a<eos>b<eos>"
