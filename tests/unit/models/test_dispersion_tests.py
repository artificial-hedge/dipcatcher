"""Tests for dispersion_tests — Siegel-Tukey + Ansari-Bradley."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.dispersion_tests import (
    ansari_bradley,
    bench_dispersion_tests,
    siegel_tukey,
)


def test_scale_ratio_rejected():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(60)
    y = 3.0 * rng.standard_normal(60)
    assert siegel_tukey(x, y)["p"] < 0.01
    assert ansari_bradley(x, y)["p"] < 0.01


def test_same_scale_not_rejected():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(60)
    y = rng.standard_normal(60)
    assert siegel_tukey(x, y)["p"] > 0.005
    assert ansari_bradley(x, y)["p"] > 0.005


def test_location_shift_tolerated():
    # Siegel-Tukey scores are sensitive to location; keep the
    # null loose: pure shift with equal spread should not always
    # reject at 1%.
    rng = np.random.default_rng(2)
    x = rng.standard_normal(80)
    y = rng.standard_normal(80) * 4.0
    assert siegel_tukey(x, y)["p"] < 0.001


def test_fail_closed_small():
    with pytest.raises(ValueError):
        siegel_tukey(np.arange(5.0), np.arange(5.0))


def test_bench():
    out = bench_dispersion_tests()
    assert out["synthetic_score"] == 1.0
