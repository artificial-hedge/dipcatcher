from quant_fund.models.chung_lil import bench_chung_lil
from quant_fund.models.glivenko_cantelli import (
    bench_glivenko_cantelli,
)
from quant_fund.models.khintchine_lln import bench_khintchine_lln
from quant_fund.models.kolmogorov_3series import (
    bench_kolmogorov_3series,
)
from quant_fund.models.levy_convergence import (
    bench_levy_convergence,
)
from quant_fund.models.strassen_lil import bench_strassen_lil


def test_strassen_lil():
    assert bench_strassen_lil()["synthetic_strassen_lil"] == 1.0


def test_chung_lil():
    assert bench_chung_lil()["synthetic_chung_lil"] == 1.0


def test_kolmogorov_3series():
    assert bench_kolmogorov_3series()["synthetic_kolmogorov_3series"] == 1.0


def test_khintchine_lln():
    assert bench_khintchine_lln()["synthetic_khintchine_lln"] == 1.0


def test_levy_convergence():
    assert bench_levy_convergence()["synthetic_levy_convergence"] == 1.0


def test_glivenko_cantelli():
    assert bench_glivenko_cantelli()["synthetic_glivenko_cantelli"] == 1.0
