"""Tests for ng_perron — Ng-Perron battery (wave-56 unit-root suite)."""

import numpy as np
import pytest

from quant_fund.models.ng_perron import bench_ng_perron, ng_perron_test, synth_ngp


def test_random_walk_decision() -> None:
    rw, _ = synth_ngp(seed=1)
    r = ng_perron_test(rw)
    rej = r["unanimous_reject"]
    assert rej == 0.0


def test_stationary_decision() -> None:
    _, st = synth_ngp(seed=2)
    r = ng_perron_test(st)
    rej = r["unanimous_reject"]
    assert rej == 1.0


def test_fail_closed() -> None:
    rw, _ = synth_ngp(seed=3)
    with pytest.raises(ValueError):
        ng_perron_test(rw[:40])
    with pytest.raises(ValueError):
        ng_perron_test(np.full(200, np.nan))
    with pytest.raises(ValueError):
        ng_perron_test(np.ones(200))


def test_determinism() -> None:
    rw, _ = synth_ngp(seed=4)
    assert ng_perron_test(rw) == ng_perron_test(rw)


def test_bench_schema_and_score() -> None:
    r = bench_ng_perron()
    for k, v in r.items():
        assert np.isfinite(v), k
    assert r["score"] == 1.0
