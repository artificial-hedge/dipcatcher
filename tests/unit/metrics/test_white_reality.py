import numpy as np
import pytest

from quant_fund.metrics.white_reality import (
    bench_white_reality,
    hansen_spa,
    synth_snooping,
    white_reality,
)


def test_rc_detects_edges() -> None:
    s = synth_snooping(true_edges=3, edge=0.25, seed=0)
    out = white_reality(s, n_boot=200, seed=1)
    assert out["p_rc"] < 0.1
    assert out["best_mean"] > 0


def test_rc_null_calibrated() -> None:
    s = synth_snooping(true_edges=0, seed=2)
    out = white_reality(s, n_boot=200, seed=3)
    assert out["p_rc"] > 0.05


def test_spa_detects_edges() -> None:
    s = synth_snooping(true_edges=2, edge=0.3, seed=4)
    out = hansen_spa(s, n_boot=200, seed=5)
    assert out["p_spa"] < 0.1
    assert out["best_t"] > 2


def test_spa_null() -> None:
    s = synth_snooping(true_edges=0, seed=6)
    out = hansen_spa(s, n_boot=200, seed=7)
    assert out["p_spa"] > 0.05


def test_validation() -> None:
    with pytest.raises(ValueError):
        white_reality(np.ones((10, 3)))
    with pytest.raises(ValueError):
        hansen_spa(np.ones((100, 3)) * np.nan)
    with pytest.raises(ValueError):
        white_reality(np.ones(60))


def test_determinism() -> None:
    s = synth_snooping(true_edges=1, seed=8)
    a = white_reality(s, n_boot=100, seed=9)
    b = white_reality(s, n_boot=100, seed=9)
    assert a["p_rc"] == b["p_rc"]


def test_bench() -> None:
    out = bench_white_reality()
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
    assert all(np.isfinite(v) for v in out.values())
