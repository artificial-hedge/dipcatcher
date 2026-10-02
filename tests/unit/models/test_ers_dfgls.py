"""Tests for ers_dfgls — ERS DF-GLS test (wave-56 unit-root suite)."""

import numpy as np
import pytest

from quant_fund.models.ers_dfgls import bench_dfgls, dfgls_test, synth_dfgls


def test_random_walk_decision() -> None:
    rw, _ = synth_dfgls(seed=1)
    r = dfgls_test(rw)
    rej = r.get("reject_unit_root")
    assert rej == 0.0


def test_stationary_decision() -> None:
    _, st = synth_dfgls(seed=2)
    r = dfgls_test(st)
    rej = r.get("reject_unit_root")
    assert rej == 1.0


def test_fail_closed() -> None:
    rw, _ = synth_dfgls(seed=3)
    with pytest.raises(ValueError):
        dfgls_test(rw[:40])
    with pytest.raises(ValueError):
        dfgls_test(np.full(200, np.nan))
    with pytest.raises(ValueError):
        dfgls_test(np.ones(200))


def test_determinism() -> None:
    rw, _ = synth_dfgls(seed=4)
    assert dfgls_test(rw) == dfgls_test(rw)


def test_bench_schema_and_score() -> None:
    r = bench_dfgls()
    for k, v in r.items():
        assert np.isfinite(v), k
    assert r["score"] == 1.0
