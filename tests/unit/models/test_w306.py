"""Unit tests for wave-306 astronomy-3/IOD canon modules."""

import numpy as np

from quant_fund.models.cowell_j2 import propagate as j2_propagate
from quant_fund.models.cr3bp_dynamics import find_l1, jacobi
from quant_fund.models.davenport_q import davenport_q, quat_to_dcm
from quant_fund.models.laplace_iod import radec_to_los
from quant_fund.models.porkchop_grid import _body_state


def test_radec_los_unit():
    v = radec_to_los(0.7, -0.4)
    assert abs(np.linalg.norm(v) - 1.0) < 1e-12
    assert v[2] < 0


def test_j2_propagation_bound():
    r0 = np.array([6800.0, 0.0, 0.0])
    v0 = np.array([0.0, 7.5, 1.0])
    r1, _ = j2_propagate(r0, v0, 60.0, nstep=30)
    assert np.linalg.norm(r1) > 6378.0


def test_l1_bracket():
    x = find_l1(0.01215)
    assert -0.01215 < x < 0.98785


def test_jacobi_conserved():
    s = np.array([0.8, 0.0, 0.0, 0.0, 0.1, 0.0])
    assert np.isfinite(jacobi(s, 0.01215))


def test_body_state_circular():
    r, v = _body_state(1.5e8, 0.3)
    assert abs(np.linalg.norm(r) - 1.5e8) < 1e-3
    assert abs(float(r @ v)) < 1e-3


def test_davenport_identity():
    r = np.eye(3)
    q = davenport_q(r, r)
    assert abs(q[0] - 1.0) < 1e-10
    np.testing.assert_allclose(quat_to_dcm(q), np.eye(3), atol=1e-10)
