from quant_fund.models.bayes_inverse import (
    bench_bayes_inverse,
)
from quant_fund.models.iter_regularize import (
    bench_iter_regularize,
)
from quant_fund.models.l_curve_opt import (
    bench_l_curve_opt,
)
from quant_fund.models.morozov_dp import (
    bench_morozov_dp,
)
from quant_fund.models.tikhonov_reg import (
    bench_tikhonov_reg,
)
from quant_fund.models.tv_denoise import (
    bench_tv_denoise,
)


def test_tikhonov_reg():
    assert bench_tikhonov_reg()["synthetic_tikhonov_reg"] == 1.0


def test_morozov_dp():
    assert bench_morozov_dp()["synthetic_morozov_dp"] == 1.0


def test_l_curve_opt():
    assert bench_l_curve_opt()["synthetic_l_curve_opt"] == 1.0


def test_iter_regularize():
    assert bench_iter_regularize()["synthetic_iter_regularize"] == 1.0


def test_tv_denoise():
    assert bench_tv_denoise()["synthetic_tv_denoise"] == 1.0


def test_bayes_inverse():
    assert bench_bayes_inverse()["synthetic_bayes_inverse"] == 1.0
