"""Unit tests for quant_fund.models._al_synth."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models._al_synth import al_loop, make_data, probs, random_baseline


def _sel(first: np.ndarray) -> object:
    def select(uP: np.ndarray, u_idx, lP, ly, w) -> np.ndarray:
        return np.asarray(first)

    return select


def test_make_data_deterministic() -> None:
    a = make_data(seed=3)
    b = make_data(seed=3)
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))


def test_al_loop_rejects_negative_index() -> None:
    # negative selection silently wrapped to the tail before
    with pytest.raises(ValueError, match="invalid unlabeled index"):
        al_loop(_sel(np.array([-1])), seed=0, rounds=1)


def test_al_loop_rejects_out_of_range_index() -> None:
    with pytest.raises(ValueError, match="invalid unlabeled index"):
        al_loop(_sel(np.array([10**9])), seed=0, rounds=1)


def test_al_loop_rejects_duplicate_selection() -> None:
    with pytest.raises(ValueError, match="invalid unlabeled index"):
        al_loop(_sel(np.array([0, 0])), seed=0, rounds=1, batch=4)


def test_al_loop_rejects_fractional_index() -> None:
    with pytest.raises(ValueError, match="invalid unlabeled index"):
        al_loop(_sel(np.array([1.5])), seed=0, rounds=1)


def test_random_baseline_deterministic_and_valid() -> None:
    a = random_baseline(seed=2, rounds=3)
    b = random_baseline(seed=2, rounds=3)
    assert a == b
    assert 0.0 <= a <= 1.0


def test_probs_bounded() -> None:
    Xp, _yp, _Xt, _yt = make_data(seed=1)
    w = np.zeros(6)
    p = probs(np.ones((5, 6)), w)
    assert np.all((p > 0) & (p < 1))
    _ = Xp
