"""Tests for kuiper."""

from __future__ import annotations

import numpy as np

from quant_fund.models.kuiper import bench_kuiper, kuiper_test, kuiper_two_sample


def test_vonmises_rejected():
    rng = np.random.default_rng(0)
    x = rng.vonmises(0.0, 1.0, 300) % (2 * np.pi)
    out = kuiper_test(np.sort(x / (2 * np.pi)))
    assert out["p"] < 0.01
    assert out["v"] > 0


def test_uniform_null_not_rejected():
    rng = np.random.default_rng(1)
    out = kuiper_test(np.sort(rng.random(300)))
    assert out["p"] > 0.01


def test_two_sample_shifted():
    rng = np.random.default_rng(2)
    a = rng.vonmises(0.0, 1.0, 200) % (2 * np.pi)
    b = rng.vonmises(1.5, 1.0, 200) % (2 * np.pi)
    out = kuiper_two_sample(a / (2 * np.pi), b / (2 * np.pi))
    assert out["p"] < 0.05


def test_bench():
    out = bench_kuiper()
    assert out["synthetic_score"] == 1.0
