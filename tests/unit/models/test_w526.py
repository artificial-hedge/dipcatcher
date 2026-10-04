from quant_fund.models.douady_hubbard import bench_douady_hubbard
from quant_fund.models.fatou_set import bench_fatou_set
from quant_fund.models.julia_set import bench_julia_set
from quant_fund.models.mandelbrot_set import bench_mandelbrot_set
from quant_fund.models.parabolic_impl import bench_parabolic_impl
from quant_fund.models.sullivan_no_wander import bench_sullivan_no_wander


def test_julia_set():
    assert bench_julia_set()["synthetic_julia_set"] == 1.0


def test_mandelbrot_set():
    assert bench_mandelbrot_set()["synthetic_mandelbrot_set"] == 1.0


def test_fatou_set():
    assert bench_fatou_set()["synthetic_fatou_set"] == 1.0


def test_sullivan_no_wander():
    assert bench_sullivan_no_wander()["synthetic_sullivan_no_wander"] == 1.0


def test_douady_hubbard():
    assert bench_douady_hubbard()["synthetic_douady_hubbard"] == 1.0


def test_parabolic_impl():
    assert bench_parabolic_impl()["synthetic_parabolic_impl"] == 1.0
