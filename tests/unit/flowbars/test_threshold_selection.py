import numpy as np
import pytest

from quant_fund.flowbars.bars import synth_tape
from quant_fund.flowbars.threshold_selection import select_threshold, threshold_profile

pytestmark = pytest.mark.synthetic


def test_profile_shapes_and_positive() -> None:
    tape = synth_tape(6000, seed=60)
    thresholds = np.array([500.0, 1000.0, 2000.0, 4000.0])
    prof = threshold_profile(tape["dollar"], tape["price"], thresholds)
    assert prof["n_bars"].shape == (4,)
    assert np.all(np.diff(prof["n_bars"]) < 0)  # higher threshold → fewer bars
    assert np.all(prof["cv"] > 0)


def test_select_threshold_in_grid() -> None:
    tape = synth_tape(8000, seed=61)
    thresholds = np.array([300.0, 700.0, 1500.0, 3000.0, 6000.0])
    out = select_threshold(tape["dollar"], tape["price"], thresholds)
    assert out["threshold"] in thresholds
    assert out["cv"] > 0
    assert out["n_bars"] > 10


def test_deterministic() -> None:
    tape = synth_tape(6000, seed=62)
    thresholds = np.array([500.0, 1500.0, 4000.0])
    a = select_threshold(tape["dollar"], tape["price"], thresholds)
    b = select_threshold(tape["dollar"], tape["price"], thresholds)
    assert a == b


def test_validation() -> None:
    with pytest.raises(ValueError):
        threshold_profile(np.array([1.0, 2.0]), np.array([1.0, 2.0]), np.array([0.0]))
    with pytest.raises(ValueError):
        threshold_profile(np.ones(5), np.ones(4), np.array([1.0]))
