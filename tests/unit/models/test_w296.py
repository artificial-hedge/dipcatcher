"""Unit tests for wave-296 robotics-4 canon modules."""

import numpy as np

from quant_fund.models.chomp import chomp
from quant_fund.models.gjk_epa import epa_depth, gjk_distance
from quant_fund.models.ilqr import ilqr
from quant_fund.models.lqr_funnel import funnel_rho, lqr_gain
from quant_fund.models.rts_smoother import kalman_filter, rts_smoother
from quant_fund.models.se3_spline import exp_se3, interpolate, log_se3


def test_lqr_gain():
    A = np.array([[0.0, 1.0], [0.0, 0.0]])
    B = np.array([[0.0], [1.0]])
    K, S = lqr_gain(A, B, np.eye(2), np.eye(1))
    assert K.shape == (1, 2) and np.all(np.linalg.eigvals(A - B @ K).real < 0.0)
    assert funnel_rho(A, B, S, K) > 0.0


def test_chomp_clears():
    xi, costs = chomp(
        np.array([0.0, 0.0]), np.array([4.0, 0.0]), 25, np.array([[2.0, 0.0, 0.4]]), iters=300
    )
    assert costs[-1] < costs[0]
    assert np.allclose(xi[0], [0.0, 0.0]) and np.allclose(xi[-1], [4.0, 0.0])


def test_gjk_disjoint():
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    far = sq + np.array([3.0, 0.0])
    assert abs(gjk_distance(sq, far) - 2.0) < 1e-9
    assert epa_depth(sq, far) == 0.0


def test_gjk_overlap():
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    ov = sq + np.array([0.5, 0.0])
    assert gjk_distance(sq, ov) == 0.0
    assert epa_depth(sq, ov) > 0.0


def test_ilqr_improves():
    def dyn(x, u):
        return np.array([x[0] + 0.1 * x[1], x[1] + 0.1 * u[0]])

    def cost(x, u, terminal=False, grads=False):
        loss = (x[0] - 1.0) ** 2 + (0.0 if terminal else 0.001 * u[0] ** 2)
        if not grads:
            return loss
        return (
            loss,
            np.array([2 * (x[0] - 1.0), 0.0]),
            np.array([0.002 * u[0]]),
            np.diag([2.0, 0.0]),
            np.zeros((1, 2)),
            np.array([[0.002]]),
        )

    xs, us, trace = ilqr(dyn, cost, np.array([0.0, 0.0]), np.zeros((40, 1)), iters=30)
    assert trace[-1] < trace[0]


def test_rts():
    rng = np.random.default_rng(0)
    F = np.array([[1.0]])
    H = np.array([[1.0]])
    Q = np.array([[0.1]])
    R = np.array([[0.5]])
    xt = np.cumsum(rng.normal(0, 0.3, 40))
    ys = xt + rng.normal(0, 0.5, 40)
    xs_f, _, _, _ = kalman_filter(ys, F, H, Q, R, np.array([0.0]), np.eye(1))
    xs_s = rts_smoother(ys, F, H, Q, R, np.array([0.0]), np.eye(1))
    assert np.mean((xs_s[:, 0] - xt) ** 2) <= np.mean((xs_f[:, 0] - xt) ** 2) * 1.05


def test_se3_roundtrip():
    xi = np.array([0.3, -0.2, 0.5, 0.1, 0.4, -0.3])
    assert np.allclose(log_se3(exp_se3(xi)), xi, atol=1e-8)
    T1, T2 = exp_se3(np.zeros(6)), exp_se3(xi)
    assert np.allclose(interpolate(T1, T2, 1.0), T2, atol=1e-9)
