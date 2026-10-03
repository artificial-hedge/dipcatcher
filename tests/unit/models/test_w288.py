"""Unit tests for wave-288 measure-theory canon modules."""

import numpy as np

from quant_fund.models.conv_prob import conv_uniform
from quant_fund.models.fubini_swap import _int_xy
from quant_fund.models.leb_integral import _lebesgue, _riemann
from quant_fund.models.leb_measure import _cantor_level, outer_measure
from quant_fund.models.radon_nikodym import rn_estimate
from quant_fund.models.weak_conv import _emp_expect


def test_cantor_measure_zero():
    assert outer_measure(_cantor_level(10)) < 0.02


def test_leb_matches_riemann():
    assert abs(_lebesgue(lambda x: x**2) - _riemann(lambda x: x**2)) < 0.01


def test_conv_triangular_peak():
    g, tri = conv_uniform()
    assert tri[200] == 1.0 or abs(tri[np.argmin(np.abs(g - 1))] - 1.0) < 0.01


def test_emp_expect():
    rng = np.random.RandomState(0)
    assert abs(_emp_expect(rng.randn(50000), lambda x: x**2) - 1.0) < 0.03


def test_fubini_simple():
    i1, i2 = _int_xy(lambda x, y: x + y, n=100)
    assert abs(i1 - 1.0) < 1e-2 and abs(i1 - i2) < 1e-9


def test_rn_uniform_ratio_one():
    rng = np.random.RandomState(1)
    mid, ratio = rn_estimate(rng.rand(60000), rng.rand(60000))
    assert np.abs(ratio[5:-5] - 1.0).mean() < 0.05
