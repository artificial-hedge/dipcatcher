"""Glivenko-Cantelli probes."""

from __future__ import annotations

import numpy as np

from quant_fund.models.uniform_lln import bench_uniform_lln, ks_gap


def test_ks_gap_accepts_external_sample():
    rng = np.random.default_rng(0)
    xs = rng.standard_normal(300)
    got = ks_gap(300, rng, xs)
    x2 = np.sort(xs)
    e1 = np.arange(1, 301) / 300
    from math import erf, sqrt

    tc = np.array([0.5 * (1 + erf(v / sqrt(2))) for v in x2])
    assert np.isclose(got, np.abs(e1 - tc).max())


def test_bench_no_vacuous_checks():
    out = bench_uniform_lln()
    assert out["synthetic_uniform_lln"] == 1.0
