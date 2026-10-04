from quant_fund.models.big_witt import bench_big_witt
from quant_fund.models.good_reduction import (
    bench_good_reduction,
)
from quant_fund.models.odeur_zarba import (
    bench_odeur_zarba,
)
from quant_fund.models.potential_reduction import (
    bench_potential_reduction,
)
from quant_fund.models.tate_curve import bench_tate_curve
from quant_fund.models.witt_len2 import bench_witt_len2


def test_witt_len2():
    assert bench_witt_len2()["synthetic_witt_len2"] == 1.0


def test_big_witt():
    assert bench_big_witt()["synthetic_big_witt"] == 1.0


def test_good_reduction():
    assert bench_good_reduction()["synthetic_good_reduction"] == 1.0


def test_potential_reduction():
    assert bench_potential_reduction()["synthetic_potential_reduction"] == 1.0


def test_tate_curve():
    assert bench_tate_curve()["synthetic_tate_curve"] == 1.0


def test_odeur_zarba():
    assert bench_odeur_zarba()["synthetic_odeur_zarba"] == 1.0
