"""Unit tests for quant_fund.models._data_synth."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models._data_synth import synth_dataset, true_margin


def test_dataset_deterministic() -> None:
    rng1 = np.random.default_rng(0)
    rng2 = np.random.default_rng(0)
    a = synth_dataset(200, rng1)
    b = synth_dataset(200, rng2)
    assert all(np.array_equal(p, q) for p, q in zip(a, b, strict=True))


def test_hard_amplification_only_distractor_dims() -> None:
    # dims 4-7 amplified x4 on hard rows; dim 3 must NOT be amplified —
    # it is the spurious feature, not a distractor. On hard rows ~90% of
    # dim 3 is overwritten by the spurious pattern (sd ~1.5); the other
    # ~10% keeps raw noise, so the mixture sd is ~1.5, not ~1.9 (x4 bug).
    x, _y, hard = synth_dataset(4000, np.random.default_rng(0), hard_frac=0.5)
    sd3 = x[hard, 3].std()
    sd4 = x[hard, 4].std()
    assert sd3 < 1.7
    assert sd4 > 3.0  # amplified distractor


def test_hard_frac_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="hard_frac"):
        synth_dataset(100, np.random.default_rng(0), hard_frac=-0.1)
    with pytest.raises(ValueError, match="hard_frac"):
        synth_dataset(100, np.random.default_rng(0), hard_frac=1.5)


def test_true_margin_sign() -> None:
    x, y, _hard = synth_dataset(500, np.random.default_rng(1))
    m = true_margin(x, y)
    assert m.shape == (500,)
    # classes are separable on average: mean margin positive
    assert m.mean() > 0
