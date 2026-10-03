"""Wave-975 distribution-theory canon tests."""

from __future__ import annotations

from quant_fund.models.dist_convolution import bench_dist_convolution
from quant_fund.models.paley_wiener import bench_paley_wiener
from quant_fund.models.schwartz_dist import bench_schwartz_dist
from quant_fund.models.sing_support import bench_sing_support
from quant_fund.models.sobolev_trace import bench_sobolev_trace
from quant_fund.models.temper_dist import bench_temper_dist


def test_schwartz_dist():
    assert bench_schwartz_dist()["synthetic_schwartz_dist"] == 1.0


def test_temper_dist():
    assert bench_temper_dist()["synthetic_temper_dist"] == 1.0


def test_dist_convolution():
    assert bench_dist_convolution()["synthetic_dist_convolution"] == 1.0


def test_sing_support():
    assert bench_sing_support()["synthetic_sing_support"] == 1.0


def test_paley_wiener():
    assert bench_paley_wiener()["synthetic_paley_wiener"] == 1.0


def test_sobolev_trace():
    assert bench_sobolev_trace()["synthetic_sobolev_trace"] == 1.0
