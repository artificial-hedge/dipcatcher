"""Probes for _peft_synth."""

import numpy as np
import pytest

from quant_fund.models._peft_synth import synth_peft_base, synth_peft_shift


def test_base_shapes_labels_deterministic():
    rng = np.random.default_rng(0)
    x, y = synth_peft_base(50, rng)
    assert x.shape == (50, 4) and y.shape == (50,)
    assert set(np.unique(y)) <= {0, 1}
    rng2 = np.random.default_rng(0)
    x2, y2 = synth_peft_base(50, rng2)
    np.testing.assert_array_equal(x, x2)
    np.testing.assert_array_equal(y, y2)


def test_shift_rot_changes_boundary():
    _, yb = synth_peft_base(200, np.random.default_rng(1))
    _, ys = synth_peft_shift(200, np.random.default_rng(1), rot=0.6)
    assert not np.array_equal(yb, ys)  # rotated boundary flips some labels


@pytest.mark.parametrize("n", [0, -7])
def test_base_hostile_n(n):
    with pytest.raises(ValueError):
        synth_peft_base(n, np.random.default_rng(0))


@pytest.mark.parametrize("n", [0, -3])
def test_shift_hostile_n(n):
    with pytest.raises(ValueError):
        synth_peft_shift(n, np.random.default_rng(0))


@pytest.mark.parametrize("rot", [np.nan, np.inf])
def test_shift_hostile_rot(rot):
    with pytest.raises(ValueError):
        synth_peft_shift(10, np.random.default_rng(0), rot=rot)
