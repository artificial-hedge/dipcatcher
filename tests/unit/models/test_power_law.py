"""Unit tests for quant_fund.models.power_law."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.power_law import bench_powerlaw, pl_fit


def _pareto(alpha: float = 2.5, n: int = 1500) -> np.ndarray:
    rng = np.random.default_rng(3)
    return (1.0 - rng.random(n)) ** (1.0 / (1.0 - alpha))


def test_pareto_recovery() -> None:
    out = pl_fit(_pareto(), n_boot=40, seed=1)
    assert abs(out["alpha"] - 2.5) < 0.3
    assert abs(out["x_min"] - 1.0) < 0.4
    assert out["p_value"] > 0.05


def test_output_fields() -> None:
    out = pl_fit(_pareto(), n_boot=20, seed=2)
    for k in ("x_min", "alpha", "ks", "p_value", "n_tail", "alpha_boot_std"):
        assert k in out
        assert np.isfinite(out[k])
    assert 0.0 <= out["p_value"] <= 1.0
    assert out["alpha"] > 1.0


def test_input_validation() -> None:
    with pytest.raises(ValueError):
        pl_fit(np.ones(10))
    with pytest.raises(ValueError):
        pl_fit(np.concatenate([np.ones(200), -np.ones(10)]))
    with pytest.raises(ValueError):
        pl_fit(np.full(200, np.nan))


def test_determinism() -> None:
    x = _pareto()
    a = pl_fit(x, n_boot=25, seed=9)
    b = pl_fit(x, n_boot=25, seed=9)
    assert a["alpha"] == b["alpha"]
    assert a["p_value"] == b["p_value"]


def test_bench_contract() -> None:
    out = bench_powerlaw()
    assert out["synthetic_score"] == 1.0
    assert out["synthetic_pl_pval_exp"] < 0.15
