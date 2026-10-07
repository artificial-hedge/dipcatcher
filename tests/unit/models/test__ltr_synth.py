"""Adversarial probes for quant_fund.models._ltr_synth."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models._ltr_synth import ltr_data, ndcg_at


def test_ltr_data_deterministic() -> None:
    a = ltr_data(seed=9, n_q=8, n_doc=5, d=4)
    b = ltr_data(seed=9, n_q=8, n_doc=5, d=4)
    for x, y in zip(a, b, strict=True):
        assert np.array_equal(x, y)


def test_ltr_data_hostile_params() -> None:
    with pytest.raises(ValueError):
        ltr_data(n_q=0)
    with pytest.raises(ValueError):
        ltr_data(n_doc=0)
    with pytest.raises(ValueError):
        ltr_data(d=0)


def test_ndcg_perfect_and_floor() -> None:
    y = np.array([[2, 1, 0, 0], [0, 2, 1, 0]])
    perfect = np.array([[3.0, 2.0, 1.0, 0.0], [0.0, 3.0, 2.0, 1.0]])
    assert ndcg_at(y, perfect, k=4) == pytest.approx(1.0)
    worst = -perfect
    assert ndcg_at(y, worst, k=4) < 0.6


def test_ndcg_tie_break_deterministic() -> None:
    # all tied scores: stable argsort must give a reproducible value
    y = np.array([[2, 0, 1, 0]])
    tied = np.zeros((1, 4))
    a = ndcg_at(y, tied, k=2)
    b = ndcg_at(y, tied, k=2)
    assert a == b


def test_ndcg_hostile_params() -> None:
    y = np.array([[1, 0]])
    s = np.array([[0.5, 0.5]])
    with pytest.raises(ValueError):
        ndcg_at(y, s, k=0)
    with pytest.raises(ValueError):
        ndcg_at(y, s, k=3)  # k > n_docs
    with pytest.raises(ValueError):
        ndcg_at(np.array([[-1, 0]]), s, k=1)  # negative grade
    with pytest.raises(ValueError):
        ndcg_at(y, np.array([0.5]), k=1)  # shape mismatch
