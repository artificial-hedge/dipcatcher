"""Tests for quant_fund.diffbacktest.grad_fidelity."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.diffbacktest.grad_fidelity import grad_fidelity, grad_fidelity_bench
from quant_fund.diffbacktest.jax_core import available
from quant_fund.diffbacktest.numpy_core import synthetic_prices

pytestmark = pytest.mark.skipif(not available(), reason="jax extra not installed")


def test_fidelity_reports_geometry():
    out = grad_fidelity(synthetic_prices(200, 5, 1), "tsmom", n_samples=4, seed=1)
    assert out["status"] == "ok"
    assert len(out["cosines"]) == 4
    assert -1.0 <= out["mean_cosine"] <= 1.0
    assert 0.0 <= out["negative_cosine_share"] <= 1.0
    assert out["mean_phantom_share"] is None or 0.0 <= out["mean_phantom_share"] <= 1.0


def test_fd_gradient_finite_and_shaped():
    from quant_fund.diffbacktest.grad_fidelity import _hard_fd_gradient
    from quant_fund.diffbacktest.spec import StrategyParams, active_parameters

    px = synthetic_prices(120, 4, 2)
    names = active_parameters("tsmom")
    g = _hard_fd_gradient(px, "tsmom", StrategyParams(), names, "sharpe", rel_step=1e-3)
    assert g.shape == (len(names),)
    assert np.isfinite(g).all()


def test_unknown_strategy_rejected():
    with pytest.raises(ValueError, match="unknown strategy"):
        grad_fidelity(synthetic_prices(64, 3, 0), "bogus", n_samples=1)


def test_bench_sealed_and_deterministic():
    r1 = grad_fidelity_bench(n_samples=6, seed=0)
    r2 = grad_fidelity_bench(n_samples=6, seed=0)
    assert r1["schema"] == "grad_fidelity.v1"
    assert r1["data_label"] == "SYNTHETIC"
    assert r1["live_pnl_claim"] is False
    assert r1 == r2
