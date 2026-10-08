"""Tests for pelt_wbs (wave-58)."""

import numpy as np
import pytest

from quant_fund.models.pelt_wbs import (
    bench_pelt_wbs,
    pelt_mean,
    synth_changepoints,
    wbs_mean,
)


def test_pelt_recovers_three() -> None:
    x, cps, _ = synth_changepoints(seed=1)
    got = pelt_mean(x, penalty=2.0 * np.log(x.size))["changepoints"]
    assert got.size == cps.size
    assert np.max(np.abs(np.sort(got) - cps)) <= 8


def test_null_no_cps() -> None:
    _, _, null = synth_changepoints(seed=2)
    got = pelt_mean(null, penalty=2.0 * np.log(null.size))["n_cp"][0]
    assert got <= 1


def test_wbs_finds_breaks() -> None:
    x, _, _ = synth_changepoints(seed=3)
    got = wbs_mean(x, n_intervals=300, threshold=2.5, min_dist=15, seed=3)["n_cp"][0]
    assert got >= 2


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        pelt_mean(np.ones(30))
    with pytest.raises(ValueError):
        wbs_mean(np.full(100, np.nan))
    with pytest.raises(ValueError):
        pelt_mean(np.ones(200), penalty=-1.0)


def test_bench_schema_and_score() -> None:
    r = bench_pelt_wbs()
    for k, v in r.items():
        assert np.isfinite(v), k
    assert r["synthetic_score"] == 1.0
