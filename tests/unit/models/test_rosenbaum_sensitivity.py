import numpy as np
import pytest

from quant_fund.models.rosenbaum_sensitivity import (
    bench_rosenbaum_sensitivity,
    sensitivity_bounds,
    synth_matched_pairs,
)


def test_gamma1_pvalue_small_for_strong_effect() -> None:
    d = synth_matched_pairs(n=250, effect=1.0, seed=0)
    out = sensitivity_bounds(np.asarray(d["diff"]))
    assert out["p_at_gamma1"] < 0.001


def test_gamma_star_shrinks_with_confounding() -> None:
    clean = sensitivity_bounds(np.asarray(synth_matched_pairs(confound=0.0, seed=1)["diff"]))
    conf = sensitivity_bounds(np.asarray(synth_matched_pairs(confound=0.7, seed=1)["diff"]))
    assert conf["gamma_star"] < clean["gamma_star"]


def test_monotone_p_upper() -> None:
    d = synth_matched_pairs(n=200, effect=0.8, seed=2)
    g = np.linspace(1.0, 4.0, 7)
    out = sensitivity_bounds(np.asarray(d["diff"]), gammas=g)
    assert out["p_upper_max"] >= out["p_at_gamma1"]


def test_zero_effect_insensitive() -> None:
    d = synth_matched_pairs(n=200, effect=0.0, seed=3)
    out = sensitivity_bounds(np.asarray(d["diff"]))
    assert out["p_at_gamma1"] > 0.1


def test_pair_count() -> None:
    d = synth_matched_pairs(n=120, effect=0.5, seed=4)
    out = sensitivity_bounds(np.asarray(d["diff"]))
    assert out["n_pairs"] == 120.0


def test_validation() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        sensitivity_bounds(rng.normal(0, 1, 5))
    with pytest.raises(ValueError):
        sensitivity_bounds(rng.normal(0, 1, 30) * np.nan)
    with pytest.raises(ValueError):
        sensitivity_bounds(np.zeros(30))
    with pytest.raises(ValueError):
        sensitivity_bounds(rng.normal(0, 1, 30), alpha=0.9)


def test_bench() -> None:
    out = bench_rosenbaum_sensitivity()
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
    assert all(np.isfinite(v) for v in out.values())
