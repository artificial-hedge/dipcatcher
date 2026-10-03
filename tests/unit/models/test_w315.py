"""Wave-315 robotics-5 module unit tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models.dmp_control import canonical, learn_dmp, rollout
from quant_fund.models.ds_motion import modulate, simulate
from quant_fund.models.grasp_epsilon import contact_wrenches, grasp_score
from quant_fund.models.rmpflow import obstacle_rmp, planar_arm_fk, planar_arm_jac
from quant_fund.models.rrt_connect import extend, seg_free
from quant_fund.models.wbc_qp import solve_prioritized


def test_fk_jac_consistent() -> None:
    q = np.array([0.3, 0.5])
    lengths = np.array([1.0, 1.0])
    j = planar_arm_jac(q, lengths)
    eps = 1e-6
    for k in range(2):
        dq = q.copy()
        dq[k] += eps
        num = (planar_arm_fk(dq, lengths) - planar_arm_fk(q, lengths)) / eps
        assert np.allclose(j[:, k], num, atol=1e-4)


def test_obstacle_rmp_cutoff() -> None:
    f, m = obstacle_rmp(np.array([0.0, 1.0]), np.zeros(2), np.zeros(2), 0.2)
    assert float(m[0, 0]) == 0.0
    f2, m2 = obstacle_rmp(np.array([0.0, 0.3]), np.zeros(2), np.zeros(2), 0.2)
    assert float(m2[0, 0]) > 0.0 and float(f2[1]) > 0.0


def test_modulate_blocks_inward() -> None:
    v = modulate(np.array([0.26, 0.0]), np.array([-1.0, 0.0]), np.zeros(2), 0.25)
    assert float(v[0]) >= -0.05


def test_simulate_converges() -> None:
    xf, min_d = simulate(np.array([0.1, 0.1]), np.array([1.2, 0.0]), np.array([0.6, 0.05]), 0.2)
    assert np.linalg.norm(xf - np.array([1.2, 0.0])) < 0.05
    assert min_d > 0.19


def test_prioritized_task1_exact() -> None:
    rng = np.random.default_rng(0)
    j1 = rng.normal(size=(2, 3))
    t1 = rng.normal(size=2)
    q = solve_prioritized([(j1, t1), (np.eye(3), rng.normal(size=3))])
    assert np.linalg.norm(j1 @ q - t1) < 1e-3


def test_contact_wrench_shape() -> None:
    w = contact_wrenches(np.array([1.0, 0.0]), np.array([-1.0, 0.0]), 0.4)
    assert w.shape == (8, 3)
    assert float(grasp_score([(np.array([-1.0, 0]), np.array([1.0, 0]))], 0.5)) == 0.0


def test_seg_free() -> None:
    obs = [(np.array([0.5, 0.5]), 0.1)]
    assert not seg_free(np.array([0.0, 0.5]), np.array([1.0, 0.5]), obs)
    assert seg_free(np.array([0.0, 0.0]), np.array([0.4, 0.0]), obs)


def test_extend_returns_new_node() -> None:
    tree = [np.array([0.0, 0.0])]
    i = extend(tree, [-1], np.array([0.5, 0.0]), [], 0.1, (0.0, 1.0))
    assert i == 1 and abs(np.linalg.norm(tree[1]) - 0.1) < 1e-9


def test_dmp_canonical() -> None:
    s = canonical(np.array([0.0, 0.5]), 1.0)
    assert s[0] == 1.0 and 0.0 < s[1] < 1.0


def test_dmp_learns_line() -> None:
    demo = np.linspace(0.0, 1.0, 100)
    w, c, h = learn_dmp(demo, 0.01, 1.0)
    traj = rollout(w, c, h, 0.0, 1.0, 200, 0.005, 1.0)
    assert abs(float(traj[-1]) - 1.0) < 0.05
