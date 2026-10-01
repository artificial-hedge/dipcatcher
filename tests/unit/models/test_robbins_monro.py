"""Stochastic approximation tests."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.robbins_monro import (
    bench_robbins_monro,
    kiefer_wolfowitz,
    robbins_monro,
    robbins_monro_averaged,
    spsa,
)


def _root_oracle(x: float, rng: np.random.Generator) -> float:
    return x - 5.0 + rng.normal(0.0, 1.5)


def _quad_oracle(x: float, rng: np.random.Generator) -> float:
    return -((x - 2.0) ** 2) + rng.normal(0.0, 0.8)


def test_rm_converges_to_root():
    out = robbins_monro(_root_oracle, 0.0, n_iter=1000, a0=1.5, seed=0)
    assert abs(float(out["root"]) - 5.0) < 0.5


def test_pr_averaging_converges():
    out = robbins_monro_averaged(_root_oracle, 0.0, n_iter=1000, a0=1.5, seed=0)
    assert abs(float(out["root"]) - 5.0) < 0.5


def test_kw_finds_argmax():
    out = kiefer_wolfowitz(_quad_oracle, -1.0, n_iter=800, a0=0.5, c0=0.6, seed=0)
    assert abs(float(out["argmax"]) - 2.0) < 0.5


def test_kw_respects_bounds():
    out = kiefer_wolfowitz(
        _quad_oracle,
        0.0,
        n_iter=500,
        a0=0.5,
        c0=0.6,
        seed=0,
        bounds=(0.0, 1.0),
    )
    assert 0.0 <= float(out["argmax"]) <= 1.0


def test_spsa_minimizes_quadratic():
    def oracle(x: np.ndarray, rng: np.random.Generator) -> float:
        return -float(np.sum(x**2)) + rng.normal(0.0, 0.5)

    out = spsa(oracle, np.array([3.0, -3.0]), n_iter=800, seed=0)
    assert float(np.linalg.norm(np.asarray(out["argmax"]))) < 1.5


def test_input_validation():
    with pytest.raises(ValueError):
        robbins_monro(_root_oracle, 0.0, a0=-1.0)
    with pytest.raises(ValueError):
        robbins_monro_averaged(_root_oracle, 0.0, power=1.0)
    with pytest.raises(ValueError):
        spsa(lambda x, rng: 0.0, np.zeros((2, 2)))


def test_bench_passes():
    out = bench_robbins_monro(seed=13)
    assert out["synthetic_rm_err"] < 0.5
    assert out["synthetic_pr_err"] < 0.5
    assert out["synthetic_kw_err"] < 0.5
    assert out["synthetic_spsa_err"] < 0.7
    assert out["synthetic_score"] == 1.0
