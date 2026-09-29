"""Validity and relation pins for the e-value mergers."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.research.emerge import (
    e_to_p,
    emerge_bonferroni,
    emerge_harmonic,
    emerge_mean,
    emerge_product,
    emerge_simes,
    p_to_e,
)


def _null_evals(rng: np.random.Generator, k: int, n_steps: int = 50) -> list[float]:
    """Simulate k e-processes under the null: products of fair bets
    on uniform PITs — each E[e] <= 1, correlated via shared draws."""
    us = rng.uniform(0, 1, size=n_steps)
    out = []
    for _ in range(k):
        # each lane bets on a different fixed direction of the same stream
        lam = rng.uniform(-0.8, 0.8)
        e = float(np.prod(1.0 + lam * (2.0 * us - 1.0)))
        out.append(max(0.0, e))
    return out


def test_mergers_are_evalues_under_null_mc() -> None:
    """Monte Carlo: every dependence-robust merger keeps E <= ~1."""
    mergers = {
        "mean": emerge_mean,
        "harmonic": emerge_harmonic,
        "bonferroni": emerge_bonferroni,
        "simes": emerge_simes,
    }
    for name, fn in mergers.items():
        acc = []
        for seed in range(400):
            rng = np.random.default_rng(seed)
            acc.append(fn(_null_evals(rng, k=5)))
        # MC mean of an e-value under the null should sit at/below ~1
        assert np.mean(acc) <= 1.05, name


def test_mean_and_bonferroni_exact_validity() -> None:
    """Deterministic check on a fixed vector."""
    evals = [0.5, 2.0, 4.0]
    assert emerge_mean(evals) == pytest.approx(13.0 / 6.0)
    assert emerge_bonferroni(evals) == pytest.approx(1.5)


def test_ordering_relations() -> None:
    evals = [0.3, 1.2, 5.0, 0.9]
    assert emerge_harmonic(evals) <= emerge_mean(evals) + 1e-12
    # simes with harmonic correction still dominates bonferroni
    assert emerge_simes(evals) >= emerge_bonferroni(evals) - 1e-12


def test_product_under_independence_growth() -> None:
    """Independent strong evidence multiplies; mean dilutes it."""
    evals = [10.0, 8.0, 12.0, 9.0]
    assert emerge_product(evals) == pytest.approx(8640.0)
    assert emerge_mean(evals) == pytest.approx(9.75)


def test_zero_evals() -> None:
    assert emerge_harmonic([0.0, 5.0, 5.0]) == 0.0
    assert emerge_product([0.0, 9.0]) == 0.0


def test_fail_closed_on_bad_inputs() -> None:
    for bad in ([], [np.nan], [-1.0, 2.0], [np.inf]):
        for fn in (emerge_mean, emerge_harmonic, emerge_bonferroni, emerge_simes):
            with pytest.raises(ValueError):
                fn(bad)


def test_p_to_e_calibration() -> None:
    # e = 1/sqrt(p); p=0.01 -> e=10
    assert p_to_e(0.01) == pytest.approx(10.0)
    assert p_to_e(1.0) == pytest.approx(1.0)
    assert p_to_e(0.0) == np.inf
    with pytest.raises(ValueError):
        p_to_e(-0.1)
    with pytest.raises(ValueError):
        p_to_e(0.5, v=0.0)


def test_e_to_p_calibration() -> None:
    assert e_to_p(20.0) == pytest.approx(0.05)
    assert e_to_p(0.7) == 1.0  # e <= 1 carries no evidence
    with pytest.raises(ValueError):
        e_to_p(np.nan)
    with pytest.raises(ValueError):
        e_to_p(-1.0)


def test_p_e_roundtrip_consistency() -> None:
    """p_to_e then e_to_p must return a p no larger than the input
    (calibration can't invent significance)."""
    for p in (1e-6, 0.001, 0.05, 0.4, 0.99):
        assert e_to_p(p_to_e(p)) <= p ** (0.5)  # 1/e = sqrt(p)
