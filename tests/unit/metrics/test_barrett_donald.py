import numpy as np
import pytest

from quant_fund.metrics.barrett_donald import (
    bench_barrett_donald,
    sd_pvalue,
    sd_statistic,
    synth_crossing,
    synth_dominance,
)


def test_dominant_direction_not_rejected() -> None:
    d = synth_dominance(shift=0.6, seed=0)
    out = sd_pvalue(np.asarray(d["x"]), np.asarray(d["y"]), n_boot=200, seed=0)
    assert out["p"] > 0.3


def test_reverse_direction_rejected() -> None:
    d = synth_dominance(shift=0.6, seed=1)
    out = sd_pvalue(np.asarray(d["y"]), np.asarray(d["x"]), n_boot=200, seed=0)
    assert out["p"] < 0.10


def test_stat_positive_when_violated() -> None:
    d = synth_dominance(shift=0.6, seed=2)
    stat = sd_statistic(np.asarray(d["y"]), np.asarray(d["x"]))
    assert stat > 1.0


def test_stat_small_when_dominant() -> None:
    d = synth_dominance(shift=0.6, seed=3)
    stat = sd_statistic(np.asarray(d["x"]), np.asarray(d["y"]))
    assert stat < 0.5


def test_second_order() -> None:
    d = synth_dominance(shift=0.4, seed=4)
    stat = sd_statistic(np.asarray(d["y"]), np.asarray(d["x"]), order=2)
    assert stat > 0


def test_crossing_rejects() -> None:
    d = synth_crossing(seed=5)
    out = sd_pvalue(np.asarray(d["x"]), np.asarray(d["y"]), n_boot=200, seed=0)
    assert out["p"] < 0.15


def test_validation() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        sd_statistic(rng.normal(0, 1, 10), rng.normal(0, 1, 10))
    with pytest.raises(ValueError):
        sd_statistic(rng.normal(0, 1, 30), rng.normal(0, 1, 30), order=3)
    with pytest.raises(ValueError):
        sd_pvalue(rng.normal(0, 1, 30), rng.normal(0, 1, 30), n_boot=10)
    with pytest.raises(ValueError):
        sd_pvalue(rng.normal(0, 1, 30) * np.nan, rng.normal(0, 1, 30))


def test_bench() -> None:
    out = bench_barrett_donald()
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
    assert all(np.isfinite(v) for v in out.values())
