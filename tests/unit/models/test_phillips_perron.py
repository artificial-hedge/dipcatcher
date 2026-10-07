"""Tests for phillips_perron — Phillips-Perron Z tests (wave-56 unit-root suite)."""

import numpy as np
import pytest

from quant_fund.models.phillips_perron import bench_phillips_perron, pp_test, synth_pp


def test_random_walk_decision() -> None:
    rw, _ = synth_pp(seed=1)
    r = pp_test(rw)
    rej = r.get("reject_unit_root")
    assert rej == 0.0


def test_stationary_decision() -> None:
    _, st = synth_pp(seed=2)
    r = pp_test(st)
    rej = r.get("reject_unit_root")
    assert rej == 1.0


def test_fail_closed() -> None:
    rw, _ = synth_pp(seed=3)
    with pytest.raises(ValueError):
        pp_test(rw[:40])
    with pytest.raises(ValueError):
        pp_test(np.full(200, np.nan))
    with pytest.raises(ValueError):
        pp_test(np.ones(200))


def test_determinism() -> None:
    rw, _ = synth_pp(seed=4)
    assert pp_test(rw) == pp_test(rw)


def test_bench_schema_and_score() -> None:
    r = bench_phillips_perron()
    for k, v in r.items():
        assert np.isfinite(v), k
    assert r["synthetic_score"] == 1.0
