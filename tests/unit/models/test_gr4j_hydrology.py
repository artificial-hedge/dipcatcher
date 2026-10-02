"""GR4J and Muskingum tests."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.gr4j_hydrology import (
    bench_hydrology,
    gr4j,
    kge,
    muskingum,
    nse,
)


def _series(seed=0, n=200):
    rng = np.random.default_rng(seed)
    P = np.clip(2 + rng.gamma(1.5, 3.0, n), 0, None)
    E = np.clip(3 + np.sin(np.arange(n) / 30.0), 0.1, None)
    return P, E


def test_gr4j_output_shape_and_nonneg():
    P, E = _series()
    q = gr4j(P, E, (300.0, 0.4, 80.0, 1.6))
    assert q.shape == P.shape
    assert (q >= 0).all()


def test_gr4j_more_rain_more_flow():
    P, E = _series()
    q_lo = gr4j(P, E, (300.0, 0.4, 80.0, 1.6))
    q_hi = gr4j(2 * P, E, (300.0, 0.4, 80.0, 1.6))
    assert q_hi[60:].mean() > q_lo[60:].mean()


def test_muskingum_mass_and_peak_attenuation():
    rng = np.random.default_rng(1)
    inflow = np.clip(rng.gamma(2, 5, 100) + np.sin(np.arange(100)) * 3 + 5, 0.1, None)
    out = muskingum(inflow, k=3.0, x=0.15)
    assert abs(out.sum() - inflow.sum()) / inflow.sum() < 0.1
    assert out.max() < inflow.max()


def test_nse_kge_perfect():
    o = np.arange(1, 50, dtype=float)
    assert nse(o, o) == pytest.approx(1.0)
    assert kge(o, o) == pytest.approx(1.0, abs=1e-9)


def test_bench_hydrology():
    out = bench_hydrology()
    assert out["synthetic_nse_self"] > 0.99
