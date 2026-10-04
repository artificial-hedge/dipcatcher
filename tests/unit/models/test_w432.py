from quant_fund.models.atiyah_hirzebruch import (
    bench_atiyah_hirzebruch,
)
from quant_fund.models.descent_ss import bench_descent_ss
from quant_fund.models.leary_ss import bench_leary_ss
from quant_fund.models.motivic_ss import bench_motivic_ss
from quant_fund.models.serre_ss3 import bench_serre_ss3
from quant_fund.models.vanishing_ss import bench_vanishing_ss


def test_atiyah_hirzebruch():
    assert bench_atiyah_hirzebruch()["synthetic_atiyah_hirzebruch"] == 1.0


def test_serre_ss3():
    assert bench_serre_ss3()["synthetic_serre_ss3"] == 1.0


def test_leary_ss():
    assert bench_leary_ss()["synthetic_leary_ss"] == 1.0


def test_descent_ss():
    assert bench_descent_ss()["synthetic_descent_ss"] == 1.0


def test_motivic_ss():
    assert bench_motivic_ss()["synthetic_motivic_ss"] == 1.0


def test_vanishing_ss():
    assert bench_vanishing_ss()["synthetic_vanishing_ss"] == 1.0
