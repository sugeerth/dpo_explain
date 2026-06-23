"""Unit tests for the pure statistics helper in evaluate.py.

`_pearson` is the only piece of evaluate.py with no model/network dependency,
and it backs the `margin_lengthdiff_corr` length control -- so it's worth
pinning down. `evaluate.py` imports torch/datasets/transformers at module
scope, so we importorskip the module rather than the formula.
"""

import math

import pytest

evaluate = pytest.importorskip(
    "evaluate", reason="evaluate.py needs torch/datasets/transformers to import"
)
_pearson = evaluate._pearson


def test_pearson_perfect_positive_correlation():
    assert _pearson([1, 2, 3, 4], [2, 4, 6, 8]) == pytest.approx(1.0)


def test_pearson_perfect_negative_correlation():
    assert _pearson([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)


def test_pearson_zero_variance_returns_zero():
    # Constant y -> denominator 0 -> defined to return 0.0 (no correlation).
    assert _pearson([1, 2, 3], [5, 5, 5]) == 0.0


def test_pearson_too_few_points_returns_zero():
    assert _pearson([1], [1]) == 0.0
    assert _pearson([], []) == 0.0


def test_pearson_matches_textbook_value():
    xs = [1.0, 2.0, 3.0, 5.0, 8.0]
    ys = [2.0, 1.0, 4.0, 3.0, 7.0]
    # Independently derived value (also matches numpy.corrcoef); not copied logic.
    assert _pearson(xs, ys) == pytest.approx(0.8766065868, abs=1e-9)


def test_pearson_is_symmetric_and_in_range():
    xs = [3.0, -1.0, 4.0, 1.0, 5.0]
    ys = [9.0, 2.0, 6.0, 5.0, 3.0]
    r = _pearson(xs, ys)
    assert _pearson(ys, xs) == pytest.approx(r)
    assert -1.0 - 1e-12 <= r <= 1.0 + 1e-12
    assert not math.isnan(r)
