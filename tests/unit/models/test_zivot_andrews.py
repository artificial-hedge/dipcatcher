"""Tests for zivot_andrews — Zivot-Andrews break test (wave-56 unit-root suite)."""

import numpy as np
import pytest

from quant_fund.models.zivot_andrews import bench_zivot_andrews, synth_za, zivot_andrews


def test_random_walk_decision() -> None:
    _, rw, _ = synth_za(seed=1)
    r = zivot_andrews(rw)
    assert r["reject_unit_root"] == 0.0


def test_stationary_decision() -> None:
    st, _, true_b = synth_za(seed=2)
    r = zivot_andrews(st)
    assert r["reject_unit_root"] == 1.0
    assert abs(r["break_idx"] - true_b) < 60


def test_fail_closed() -> None:
    _, rw, _ = synth_za(seed=3)
    with pytest.raises(ValueError):
        zivot_andrews(rw[:40])
    with pytest.raises(ValueError):
        zivot_andrews(np.full(200, np.nan))
    with pytest.raises(ValueError):
        zivot_andrews(np.ones(200))


def test_determinism() -> None:
    _, rw, _ = synth_za(seed=4)
    assert zivot_andrews(rw) == zivot_andrews(rw)


def test_bench_schema_and_score() -> None:
    r = bench_zivot_andrews()
    for k, v in r.items():
        assert np.isfinite(v), k
    assert r["synthetic_score"] == 1.0


def test_k_must_leave_dof() -> None:
    _, rw, _ = synth_za(seed=5)
    with pytest.raises(ValueError, match="k"):
        zivot_andrews(rw, k=200)
    with pytest.raises(ValueError, match="k"):
        zivot_andrews(rw, k=-1)
    with pytest.raises(ValueError, match="k"):
        zivot_andrews(rw, k=1.5)
