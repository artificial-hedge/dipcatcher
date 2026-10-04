from quant_fund.models.hensel_lift import bench_hensel_lift
from quant_fund.models.poly_crt import bench_poly_crt
from quant_fund.models.poly_eval_interp import bench_poly_eval_interp
from quant_fund.models.poly_factor_fp import bench_poly_factor_fp
from quant_fund.models.sparse_interp import bench_sparse_interp
from quant_fund.models.subresultant import bench_subresultant


def test_poly_factor_fp():
    assert bench_poly_factor_fp()["synthetic_poly_factor_fp"] == 1.0


def test_hensel_lift():
    assert bench_hensel_lift()["synthetic_hensel_lift"] == 1.0


def test_poly_crt():
    assert bench_poly_crt()["synthetic_poly_crt"] == 1.0


def test_subresultant():
    assert bench_subresultant()["synthetic_subresultant"] == 1.0


def test_sparse_interp():
    assert bench_sparse_interp()["synthetic_sparse_interp"] == 1.0


def test_poly_eval_interp():
    assert bench_poly_eval_interp()["synthetic_poly_eval_interp"] == 1.0
