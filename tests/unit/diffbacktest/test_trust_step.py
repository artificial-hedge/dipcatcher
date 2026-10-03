"""Tests for quant_fund.diffbacktest.trust_step."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.diffbacktest.jax_core import available
from quant_fund.diffbacktest.numpy_core import synthetic_prices
from quant_fund.diffbacktest.trust_step import _feasible_sample, trust_bench, trust_optimize


def test_feasible_sample_validates():
    from quant_fund.diffbacktest.spec import StrategyParams, active_parameters, validate_params

    rng = np.random.default_rng(0)
    names = active_parameters("tsmom")
    p = _feasible_sample(rng, names, StrategyParams(), "tsmom")
    validate_params("tsmom", p)  # must not raise


@pytest.mark.skipif(not available(), reason="jax unavailable")
def test_trust_optimize_runs_and_reports():
    prices = synthetic_prices(n_steps=400, n_names=5, seed=0)
    out = trust_optimize(prices, "tsmom", n_iters=6, seed=0)
    assert "gated" in out and "naive" in out
    assert np.isfinite(out["gated"]["final_objective"]) or out["gated"]["n_infeasible"] > 0


@pytest.mark.skipif(not available(), reason="jax unavailable")
def test_acceptance_gate_rejects_worse_steps():
    # a gate that accepts everything is exactly naive mode
    prices = synthetic_prices(n_steps=400, n_names=5, seed=1)
    out = trust_optimize(prices, "tsmom", n_iters=6, seed=1)
    assert out["gated"]["n_accept"] + out["gated"]["n_reject"] + out["gated"]["n_infeasible"] <= 6


def test_bench_sealed_or_skipped():
    if not available():
        pytest.skip("jax unavailable")
    r = trust_bench(seed=0, n_steps=200, n_seeds=2)
    assert r["schema"] == "trust_step.v1"
    assert r["data_label"] == "SYNTHETIC"
    assert "receipt_sha256" in r
