"""Probes for _oc_synth."""

import numpy as np
import pytest

from quant_fund.models._oc_synth import HORIZON, dyn, pd_baseline, rollout


def test_rollout_shapes_and_cost():
    xs, cost = rollout(np.zeros(10), np.array([0.0, 0.0]), target=1.0)
    assert xs.shape == (11, 2)
    assert cost > 0
    assert np.isfinite(xs).all() and np.isfinite(cost)


def test_rollout_deterministic():
    u = np.linspace(-1, 1, 15)
    a = rollout(u, np.array([0.5, -0.2]))
    b = rollout(u, np.array([0.5, -0.2]))
    np.testing.assert_array_equal(a[0], b[0])
    assert a[1] == b[1]


@pytest.mark.parametrize(
    "u,x0",
    [
        (np.zeros((3, 3)), np.zeros(2)),  # 2-D control seq
        (np.array([0.0, np.inf]), np.zeros(2)),  # non-finite control
        (np.zeros(3), np.zeros(3)),  # wrong x0 shape
        (np.zeros(3), np.array([np.nan, 0.0])),  # non-finite x0
    ],
)
def test_rollout_hostile(u, x0):
    with pytest.raises(ValueError):
        rollout(u, x0)


def test_empty_u_seq_returns_x0_only():
    xs, cost = rollout(np.zeros(0), np.array([0.3, 0.1]))
    assert xs.shape == (1, 2)
    assert cost == 0.0


def test_pd_baseline_horizon_and_converges():
    xs, cost = pd_baseline(np.array([0.0, 0.0]), target=1.0)
    assert xs.shape == (HORIZON + 1, 2)
    assert cost > 0
    # PD controller should move position toward target
    assert xs[-1, 0] > 0.5


def test_pd_baseline_rejects_bad_x0():
    with pytest.raises(ValueError):
        pd_baseline(np.zeros(4))


def test_dyn_consistency_with_jac():
    from quant_fund.models._oc_synth import dyn_jac

    x = np.array([0.4, -0.7])
    fx, fu = dyn_jac(x, 0.3)
    assert np.allclose(fx, [[1.0, 0.1], [0.0, 1.0]])
    assert np.allclose(fu, [0.0, 0.1])
    assert dyn(x, 0.0)[0] == pytest.approx(x[0] + 0.1 * x[1])
