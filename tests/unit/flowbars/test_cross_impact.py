import numpy as np
import pytest

from quant_fund.flowbars.cross_impact import (
    cross_impact_bootstrap_p,
    cross_impact_matrix,
    diagonal_dominance,
)

pytestmark = pytest.mark.synthetic


def _coupled_flows(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Two assets whose prices respond to own and peer flow."""
    rng = np.random.default_rng(seed)
    f1 = rng.standard_normal(n) * 10.0
    f2 = rng.standard_normal(n) * 10.0
    dp1 = 1e-3 * f1 + 5e-4 * f2 + rng.standard_normal(n) * 0.02
    dp2 = 5e-4 * f1 + 1e-3 * f2 + rng.standard_normal(n) * 0.02
    dp = np.column_stack([dp1, dp2])
    f = np.column_stack([f1, f2])
    return dp, f


def test_recovers_impact_matrix() -> None:
    n = 6000
    dp, f = _coupled_flows(n, seed=30)
    out = cross_impact_matrix(dp, f)
    lam = out["impact"]
    assert isinstance(lam, np.ndarray)
    assert lam.shape == (2, 2)
    assert lam[0, 0] == pytest.approx(1e-3, rel=0.2)
    assert lam[0, 1] == pytest.approx(5e-4, rel=0.3)
    assert lam[1, 0] == pytest.approx(5e-4, rel=0.3)
    assert lam[1, 1] == pytest.approx(1e-3, rel=0.2)
    r2 = out["r2"]
    assert isinstance(r2, np.ndarray)
    assert np.all(r2 > 0.15)


def test_diagonal_dominance_positive_when_own_flow_leads() -> None:
    dp, f = _coupled_flows(6000, seed=31)
    lam = cross_impact_matrix(dp, f)["impact"]
    assert isinstance(lam, np.ndarray)
    assert diagonal_dominance(lam) > 0.0


def test_bootstrap_rejects_zero_cross_impact() -> None:
    rng = np.random.default_rng(32)
    n = 4000
    # independent flows, no cross response → cross effects are pure noise
    f = rng.standard_normal((n, 2)) * 10.0
    dp = rng.standard_normal((n, 2)) * 0.02
    out = cross_impact_bootstrap_p(dp, f, block=10, n_boot=200, seed=0)
    assert out["p"] > 0.05


def test_bootstrap_detects_real_cross_impact() -> None:
    n = 6000
    dp, f = _coupled_flows(n, seed=33)
    out = cross_impact_bootstrap_p(dp, f, block=10, n_boot=200, seed=0)
    assert out["p"] < 0.05


def test_validation() -> None:
    with pytest.raises(ValueError):
        cross_impact_matrix(np.zeros((5, 3)), np.zeros((5, 3)))
    with pytest.raises(ValueError):
        diagonal_dominance(np.zeros((3, 2)))
