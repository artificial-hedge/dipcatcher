"""Brinson attribution tests."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.brinson_attribution import (
    bench_brinson,
    brinson_single,
    carino_link,
)


def test_bhb_identity_single_period():
    wp = np.array([0.5, 0.3, 0.2])
    wb = np.array([0.4, 0.4, 0.2])
    rp = np.array([0.05, 0.02, -0.01])
    rb = np.array([0.03, 0.04, 0.0])
    out = brinson_single(wp, wb, rp, rb)
    tot = out["allocation"].sum() + out["selection"].sum() + out["interaction"].sum()
    assert tot == pytest.approx(out["active_return"].item(), abs=1e-9)


def test_bhb_identity_multi_period():
    rng = np.random.default_rng(0)
    wb = np.tile([0.5, 0.5], (4, 1))
    wp = np.clip(wb + rng.normal(0, 0.05, (4, 2)), 0, None)
    wp /= wp.sum(1, keepdims=True)
    rb = rng.normal(0.01, 0.02, (4, 2))
    rp = rb + rng.normal(0.001, 0.01, (4, 2))
    out = brinson_single(wp, wb, rp, rb)
    assert np.allclose(
        out["allocation"].sum(1) + out["selection"].sum(1) + out["interaction"].sum(1),
        out["active_return"],
        atol=1e-9,
    )


def test_carino_link_finite():
    rp = np.array([0.02, -0.01, 0.015])
    rb = np.array([0.01, 0.0, 0.01])
    eff = np.array([[0.005, 0.004, 0.001], [0.0, -0.01, -0.0], [0.003, 0.002, 0.0]])
    linked = carino_link(rp, rb, eff)
    assert np.isfinite(linked).all()
    assert linked.size == 3


def test_bench_brinson():
    out = bench_brinson()
    assert out["synthetic_identity_err"] < 1e-9
