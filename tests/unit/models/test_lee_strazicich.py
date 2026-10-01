"""Tests for lee_strazicich — Lee-Strazicich LM break test (wave-56 unit-root suite)."""

import numpy as np
import pytest

from quant_fund.models.lee_strazicich import bench_lee_strazicich, lee_strazicich, synth_ls


def test_random_walk_decision() -> None:
    _, rw, _ = synth_ls(seed=1)
    r = lee_strazicich(rw)
    assert r["reject_unit_root"] == 0.0


def test_stationary_decision() -> None:
    st, _, true_b = synth_ls(seed=2)
    r = lee_strazicich(st)
    assert r["reject_unit_root"] == 1.0
    assert abs(r["break_idx"] - true_b) < 60


def test_fail_closed() -> None:
    _, rw, _ = synth_ls(seed=3)
    with pytest.raises(ValueError):
        lee_strazicich(rw[:40])
    with pytest.raises(ValueError):
        lee_strazicich(np.full(200, np.nan))
    with pytest.raises(ValueError):
        lee_strazicich(np.ones(200))


def test_determinism() -> None:
    _, rw, _ = synth_ls(seed=4)
    assert lee_strazicich(rw) == lee_strazicich(rw)


def test_bench_schema_and_score() -> None:
    r = bench_lee_strazicich()
    for k, v in r.items():
        assert np.isfinite(v), k
    assert r["score"] == 1.0
