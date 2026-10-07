"""Unit tests for quant_fund.models.chen_tiao_outliers."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.chen_tiao_outliers import (
    bench_outliers,
    outlier_scan,
    synth_outliers,
)


def test_detects_ao_near_plant() -> None:
    y, _ = synth_outliers(seed=1)
    n = y.size
    hits = [h for h in outlier_scan(y)["hits"] if h["type"] == "AO"]
    assert any(abs(int(h["tau"]) - n // 3) <= 3 for h in hits)


def test_detects_ls_near_plant() -> None:
    y, _ = synth_outliers(seed=2)
    n = y.size
    hits = list(outlier_scan(y)["hits"])
    assert any(h["type"] in ("LS", "TC") and abs(int(h["tau"]) - 2 * n // 3) <= 4 for h in hits)


def test_clean_series_no_calls() -> None:
    _, clean = synth_outliers(seed=3)
    assert len(outlier_scan(clean)["hits"]) == 0


def test_input_validation() -> None:
    with pytest.raises(ValueError):
        outlier_scan(np.ones(30))
    with pytest.raises(ValueError):
        outlier_scan(np.full(100, 2.0))


def test_bench_contract() -> None:
    out = bench_outliers()
    assert out["synthetic_score"] == 1.0
    assert out["synthetic_clean_hits"] == 0.0
