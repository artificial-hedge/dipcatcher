"""Tests for ALiBi attention bench (models/alibi_attn.py)."""

import pytest


def test_train_and_eval_deterministic_per_seed():
    pytest.importorskip("torch")
    from quant_fund.models.alibi_attn import _train_and_eval

    a = _train_and_eval(11, True, iters=4)
    b = _train_and_eval(11, True, iters=4)
    c = _train_and_eval(11, True, iters=4)
    assert a == b == c


def test_train_and_eval_seed_changes_data():
    pytest.importorskip("torch")
    from quant_fund.models.alibi_attn import _train_and_eval

    # Different seeds give different training streams; identical seeds must
    # reproduce bit-for-bit (the defect class: unseeded per-iter batch RNG).
    a = _train_and_eval(5, False, iters=4)
    b = _train_and_eval(5, False, iters=4)
    assert a == b
