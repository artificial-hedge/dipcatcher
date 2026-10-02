"""Tests for contingency — exact/conditional table tests."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.contingency import (
    bench_contingency,
    cmh,
    cramers_v,
    fisher_exact,
    mcnemar,
    phi_coefficient,
)


def test_fisher_strong_association():
    t = np.array([[35.0, 5.0], [8.0, 32.0]])
    out = fisher_exact(t)
    assert out["p"] < 0.001
    assert out["odds_ratio"] > 5.0


def test_fisher_null():
    t = np.array([[18.0, 22.0], [20.0, 20.0]])
    out = fisher_exact(t)
    assert out["p"] > 0.05


def test_mcnemar_exact():
    out = mcnemar(np.array([[10.0, 20.0], [5.0, 15.0]]))
    assert out["p"] < 0.01


def test_cmh_common_odds():
    tables = [np.array([[25.0, 5.0], [10.0, 20.0]]) for _ in range(3)]
    out = cmh(tables)
    assert out["p"] < 0.001


def test_phi_cramers():
    t = np.array([[30.0, 10.0], [10.0, 30.0]])
    assert abs(phi_coefficient(t)) > 0.4
    assert cramers_v(t) > 0.3


def test_fail_closed_shape():
    with pytest.raises(ValueError):
        fisher_exact(np.ones((3, 3)))


def test_bench():
    out = bench_contingency()
    assert out["score"] == 1.0
