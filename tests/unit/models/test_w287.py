"""Unit tests for wave-287 differential-geometry canon modules."""

import numpy as np

from quant_fund.models.christoffel import _g_polar, christoffel
from quant_fund.models.first_ff import _num_diff, _sphere
from quant_fund.models.gauss_curve import _k_num
from quant_fund.models.gauss_curve import _sphere as _sphere25
from quant_fund.models.geodesic_sphere import _to_xyz
from quant_fund.models.surf_area import _quad_area, _sphere_met


def test_ff_sphere():
    du, dv = _num_diff(_sphere, (1.0, 0.5))
    assert abs(du @ du - 4.0) < 1e-3 and abs(du @ dv) < 1e-6


def test_gauss_sphere_const():
    assert abs(_k_num(_sphere25, (1.2, 0.7)) - 1 / 6.25) < 1e-2


def test_frenet_circle():
    # helix with b=0 -> circle radius 2: kappa = 1/2
    t = 0.7
    import quant_fund.models.frenet_frame as ff

    k, _ = ff._frame(t)
    assert abs(k - 2.0 / 4.49) < 1e-3


def test_christoffel_polar():
    g = christoffel(_g_polar, (2.0, 0.5))
    assert abs(g[0, 1, 1] + 2.0) < 1e-4 and abs(g[1, 0, 1] - 0.5) < 1e-4


def test_to_xyz_unit():
    assert abs(np.linalg.norm(_to_xyz(0.7, 1.3)) - 1.0) < 1e-12


def test_surf_sphere():
    assert abs(_quad_area(_sphere_met) - 16 * np.pi) / (16 * np.pi) < 5e-3
