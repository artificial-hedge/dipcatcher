from quant_fund.models.anderson_mixing import (
    bench_anderson_mixing,
)
from quant_fund.models.conjugate_grad_ls import (
    bench_conjugate_grad_ls,
)
from quant_fund.models.gauss_newton import (
    bench_gauss_newton,
)
from quant_fund.models.landweber_iter import (
    bench_landweber_iter,
)
from quant_fund.models.levenberg_marq import (
    bench_levenberg_marq,
)
from quant_fund.models.moore_penrose import (
    bench_moore_penrose,
)


def test_moore_penrose():
    assert bench_moore_penrose()["synthetic_moore_penrose"] == 1.0


def test_landweber_iter():
    assert bench_landweber_iter()["synthetic_landweber_iter"] == 1.0


def test_conjugate_grad_ls():
    assert bench_conjugate_grad_ls()["synthetic_conjugate_grad_ls"] == 1.0


def test_gauss_newton():
    assert bench_gauss_newton()["synthetic_gauss_newton"] == 1.0


def test_levenberg_marq():
    assert bench_levenberg_marq()["synthetic_levenberg_marq"] == 1.0


def test_anderson_mixing():
    assert bench_anderson_mixing()["synthetic_anderson_mixing"] == 1.0
