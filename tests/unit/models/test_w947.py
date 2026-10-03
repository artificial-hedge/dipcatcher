"""Wave-947 spectral-interlacing canon tests."""

from __future__ import annotations

from quant_fund.models.bezout_matrix import bench_bezout_matrix
from quant_fund.models.cauchy_interlace import bench_cauchy_interlace
from quant_fund.models.haynsworth_inertia import bench_haynsworth_inertia
from quant_fund.models.min_max_eig import bench_min_max_eig
from quant_fund.models.sturm_sequence import bench_sturm_sequence
from quant_fund.models.sylvester_law import bench_sylvester_law


def test_cauchy_interlace():
    assert bench_cauchy_interlace()["synthetic_cauchy_interlace"] == 1.0


def test_sylvester_law():
    assert bench_sylvester_law()["synthetic_sylvester_law"] == 1.0


def test_haynsworth_inertia():
    assert bench_haynsworth_inertia()["synthetic_haynsworth_inertia"] == 1.0


def test_min_max_eig():
    assert bench_min_max_eig()["synthetic_min_max_eig"] == 1.0


def test_sturm_sequence():
    assert bench_sturm_sequence()["synthetic_sturm_sequence"] == 1.0


def test_bezout_matrix():
    assert bench_bezout_matrix()["synthetic_bezout_matrix"] == 1.0
