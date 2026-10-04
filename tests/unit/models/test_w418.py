from quant_fund.models.borel_cantelli import bench_borel_cantelli
from quant_fund.models.clt_classic import bench_clt_classic
from quant_fund.models.dominated_conv import bench_dominated_conv
from quant_fund.models.strong_lln import bench_strong_lln
from quant_fund.models.uniform_lln import bench_uniform_lln
from quant_fund.models.weak_law import bench_weak_law


def test_weak_law():
    assert bench_weak_law()["synthetic_weak_law"] == 1.0


def test_strong_lln():
    assert bench_strong_lln()["synthetic_strong_lln"] == 1.0


def test_clt_classic():
    assert bench_clt_classic()["synthetic_clt_classic"] == 1.0


def test_borel_cantelli():
    assert bench_borel_cantelli()["synthetic_borel_cantelli"] == 1.0


def test_dominated_conv():
    assert bench_dominated_conv()["synthetic_dominated_conv"] == 1.0


def test_uniform_lln():
    assert bench_uniform_lln()["synthetic_uniform_lln"] == 1.0
