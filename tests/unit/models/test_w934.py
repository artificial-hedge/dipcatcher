"""Wave-934 convex-analysis-2 canon tests."""

from __future__ import annotations

from quant_fund.models.inf_convolution import bench_inf_convolution
from quant_fund.models.legendre_transform import bench_legendre_transform
from quant_fund.models.normal_cone import bench_normal_cone
from quant_fund.models.perspective_fn import bench_perspective_fn
from quant_fund.models.polar_cone import bench_polar_cone
from quant_fund.models.support_fn import bench_support_fn


def test_inf_convolution():
    assert bench_inf_convolution()["synthetic_inf_convolution"] == 1.0


def test_legendre_transform():
    assert bench_legendre_transform()["synthetic_legendre_transform"] == 1.0


def test_support_fn():
    assert bench_support_fn()["synthetic_support_fn"] == 1.0


def test_perspective_fn():
    assert bench_perspective_fn()["synthetic_perspective_fn"] == 1.0


def test_polar_cone():
    assert bench_polar_cone()["synthetic_polar_cone"] == 1.0


def test_normal_cone():
    assert bench_normal_cone()["synthetic_normal_cone"] == 1.0
