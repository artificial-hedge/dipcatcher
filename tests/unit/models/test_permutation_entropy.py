"""Unit tests for quant_fund.models.permutation_entropy."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.permutation_entropy import (
    _pattern_counts,
    bench_permutation_entropy,
    complexity_entropy,
    permutation_entropy,
)


def test_pattern_counts_sum() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal(1000)
    counts = _pattern_counts(x, 4, 1)
    assert counts.sum() == 1000 - 3


def test_pe_bounds() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal(3000)
    h = permutation_entropy(x, m=5)
    assert 0.0 <= h <= 1.0


def test_pe_regime_ordering() -> None:
    rng = np.random.default_rng(2)
    n = 4000
    noise = rng.standard_normal(n)
    periodic = np.sin(np.linspace(0, 150 * np.pi, n))
    assert permutation_entropy(periodic, 5) < permutation_entropy(noise, 5)


def test_pe_deterministic() -> None:
    rng = np.random.default_rng(3)
    x = rng.standard_normal(500)
    assert permutation_entropy(x, 5) == permutation_entropy(x, 5)


def test_pe_rejects_bad_spec() -> None:
    x = np.arange(50.0)
    with pytest.raises(ValueError):
        permutation_entropy(x, m=1)
    with pytest.raises(ValueError):
        permutation_entropy(x, m=8)
    with pytest.raises(ValueError):
        permutation_entropy(np.arange(6.0), m=5)


def test_complexity_entropy_bounds() -> None:
    rng = np.random.default_rng(4)
    x = rng.standard_normal(4000)
    h, c = complexity_entropy(x, m=5)
    assert 0.0 <= h <= 1.0
    assert 0.0 <= c <= 1.0
    # noise has low complexity
    assert c < 0.2


def test_bench_permutation_entropy_score() -> None:
    out = bench_permutation_entropy()
    assert out["score"] == pytest.approx(1.0)
