"""Unit tests for wave-300 astronomy-2 canon modules."""

import numpy as np

from quant_fund.models.delta_t import delta_t
from quant_fund.models.eclipse_circ import min_separation, separation
from quant_fund.models.equinox_prec import precess, precession_matrix
from quant_fund.models.nutation_lite import nutation
from quant_fund.models.planet_vsop import _kepler_E, planet_pos
from quant_fund.models.rise_set import altitude, rise_transit_set, sun_ra_dec


def test_precession_orthogonal():
    P = precession_matrix(0.6)
    assert abs(np.linalg.det(P) - 1.0) < 1e-10
    assert np.linalg.norm(P.T @ P - np.eye(3)) < 1e-10
    assert np.linalg.norm(precession_matrix(0.0) - np.eye(3)) < 1e-12


def test_precess_unit_vector():
    v = precess(np.array([1.0, 0.0, 0.0]), 0.5)
    assert abs(np.linalg.norm(v) - 1.0) < 1e-12


def test_nutation_bounds():
    for T in (-0.5, 0.0, 0.5, 1.0):
        dp, de = nutation(T)
        assert abs(dp) < 0.006 and abs(de) < 0.004


def test_rise_set_order():
    r, t, s = rise_transit_set(2459500.5, 40.0, -74.0)
    assert r is not None and s is not None
    assert r < t < s
    assert altitude(t, 40.0, -74.0) > 0.0


def test_sun_dec_range():
    _, dec = sun_ra_dec(2459500.0)
    assert abs(dec) < 24.5


def test_eclipse_min_sep():
    jd, sep = min_separation(2451545.0, 2451546.0, 0.0, 0.0)
    assert 2451545.0 <= jd <= 2451546.0
    assert sep == separation(jd, 0.0, 0.0)


def test_delta_t_anchors():
    assert abs(delta_t(2000.0) - 63.8) < 1.5
    assert abs(delta_t(1980.0) - 50.5) < 1.5


def test_kepler_residual():
    for e in (0.01, 0.1, 0.3):
        M = 1.3
        E = _kepler_E(M, e)
        assert abs(E - e * np.sin(E) - M) < 1e-12


def test_planet_radius_bounds():
    r = planet_pos("earth", 2451545.0)
    assert 0.95 < np.linalg.norm(r) < 1.05
    rj = planet_pos("jupiter", 2455000.0)
    assert 4.0 < np.linalg.norm(rj) < 6.0
