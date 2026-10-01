"""Unit tests for quant_fund.models.bfast."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.bfast import bench_bfast, bfast_detect, synth_bfast


def test_detects_planted_break() -> None:
    y, _ = synth_bfast(seed=1)
    r = bfast_detect(y)
    assert float(r["n_breaks"]) >= 1.0
    breaks = np.asarray(r["breaks"])
    assert min(abs(b - y.size // 2) for b in breaks) <= 8


def test_clean_series_no_break() -> None:
    _, clean = synth_bfast(seed=2)
    assert float(bfast_detect(clean)["n_breaks"]) == 0.0


def test_fit_improves_ssr() -> None:
    y, _ = synth_bfast(seed=3)
    r = bfast_detect(y)
    assert float(r["ssr_final"]) < 0.8 * float(r["ssr0"])


def test_input_validation() -> None:
    with pytest.raises(ValueError):
        bfast_detect(np.ones(20))
    with pytest.raises(ValueError):
        bfast_detect(np.full(100, 3.0))


def test_bench_contract() -> None:
    out = bench_bfast()
    assert out["score"] == 1.0
    assert out["synthetic_bf_clean_breaks"] == 0.0
