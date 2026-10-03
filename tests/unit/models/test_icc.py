"""Tests for icc — intraclass correlation."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.icc import bench_icc, icc


def test_high_agreement_high_icc():
    rng = np.random.default_rng(0)
    truth = rng.standard_normal(50) * 2
    x = truth[:, None] + 0.3 * rng.standard_normal((50, 4))
    out = icc(x)
    assert out["icc_2_1"] > 0.85


def test_noise_low_icc():
    rng = np.random.default_rng(1)
    out = icc(rng.standard_normal((50, 4)))
    assert out["icc_2_1"] < 0.4


def test_all_forms_present():
    rng = np.random.default_rng(2)
    out = icc(rng.standard_normal((40, 3)))
    for k in ("icc_1_1", "icc_2_1", "icc_3_1", "icc_a_1", "icc_c_1", "sem"):
        assert np.isfinite(out[k])


def test_sem_decreases_with_agreement():
    rng = np.random.default_rng(3)
    truth = rng.standard_normal(50)
    hi = icc(truth[:, None] + 0.2 * rng.standard_normal((50, 4)))
    lo = icc(truth[:, None] + 1.5 * rng.standard_normal((50, 4)))
    assert hi["sem"] < lo["sem"]


def test_fail_closed_single_rater():
    with pytest.raises(ValueError):
        icc(np.ones((30, 1)))


def test_bench():
    out = bench_icc()
    assert out["score"] == 1.0
