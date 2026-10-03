"""Unit tests for wave-295 astronomy/orbital-mechanics canon modules."""

import numpy as np

from quant_fund.models.gauss_iod import gibbs
from quant_fund.models.kepler_solve import kepler_E
from quant_fund.models.lambert_problem import lambert
from quant_fund.models.orbit_maneuver import bielliptic, hohmann
from quant_fund.models.orbital_elements import elements_to_state, state_to_elements
from quant_fund.models.tle_propagate import secular_rates

_MU = 398600.4418


def test_roundtrip():
    a, e, inc = 12000.0, 0.3, 0.7
    r, v = elements_to_state(a, e, inc, 1.1, 0.4, 2.0, _MU)
    a2, e2, i2, _, _, _ = state_to_elements(r, v, _MU)
    assert abs(a2 - a) / a < 1e-10 and abs(e2 - e) < 1e-10 and abs(i2 - inc) < 1e-10


def test_kepler():
    E = kepler_E(1.0, 0.5)
    assert abs(E - 0.5 * np.sin(E) - 1.0) < 1e-12


def test_lambert_reaches():
    r1, _ = elements_to_state(10000.0, 0.1, 0.3, 0.3, 0.2, 0.5, _MU)
    r2, _ = elements_to_state(10000.0, 0.1, 0.3, 0.3, 0.2, 1.7, _MU)
    v1, v2 = lambert(r1, r2, 4000.0, _MU, tm=1)
    assert np.linalg.norm(v1) > 0.0 and np.linalg.norm(v2) > 0.0
    # energy of transfer orbit is negative (elliptic)
    assert np.dot(v1, v1) / 2.0 - _MU / np.linalg.norm(r1) < 0.0


def test_secular_sign():
    n = np.sqrt(_MU / 7200.0**3)
    d_raan, _, _ = secular_rates(7200.0, 0.02, np.radians(55.0), n)
    assert d_raan < 0.0  # prograde regresses


def test_hohmann():
    dv1, dv2, tof = hohmann(7000.0, 14000.0, _MU)
    assert dv1 > 0.0 and dv2 > 0.0 and tof > 0.0
    a_t = 0.5 * 21000.0
    assert abs(tof - np.pi * np.sqrt(a_t**3 / _MU)) / tof < 1e-12


def test_bielliptic_finite():
    d1, d2, d3, t = bielliptic(7000.0, 70000.0, 200000.0, _MU)
    assert all(np.isfinite(x) for x in (d1, d2, d3, t))


def test_gibbs():
    rs, vs = [], []
    for nu in (0.5, 0.9, 1.4):
        r, v = elements_to_state(11000.0, 0.2, 0.5, 0.7, 0.3, nu, _MU)
        rs.append(r)
        vs.append(v)
    assert np.linalg.norm(gibbs(*rs, _MU) - vs[1]) / np.linalg.norm(vs[1]) < 1e-8
