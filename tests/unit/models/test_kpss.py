"""Tests for kpss — KPSS stationarity test (wave-56 unit-root suite)."""

import numpy as np
import pytest

from quant_fund.models.kpss import bench_kpss, kpss_test, synth_kpss


def test_random_walk_decision() -> None:
    rw, _ = synth_kpss(seed=1)
    r = kpss_test(rw)
    rej = r.get("reject_stationarity")
    assert rej == 1.0


def test_stationary_decision() -> None:
    _, st = synth_kpss(seed=2)
    r = kpss_test(st)
    rej = r.get("reject_stationarity")
    assert rej == 0.0


def test_fail_closed() -> None:
    rw, _ = synth_kpss(seed=3)
    with pytest.raises(ValueError):
        kpss_test(rw[:40])
    with pytest.raises(ValueError):
        kpss_test(np.full(200, np.nan))
    with pytest.raises(ValueError):
        kpss_test(np.ones(200))


def test_determinism() -> None:
    rw, _ = synth_kpss(seed=4)
    assert kpss_test(rw) == kpss_test(rw)


def test_bench_schema_and_score() -> None:
    r = bench_kpss()
    for k, v in r.items():
        assert np.isfinite(v), k
    assert r["score"] == 1.0
