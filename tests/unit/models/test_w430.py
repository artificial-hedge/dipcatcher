from quant_fund.models.brane_tensor import bench_brane_tensor
from quant_fund.models.delooping import bench_delooping
from quant_fund.models.e_n_algebra import bench_e_n_algebra
from quant_fund.models.module_cat import bench_module_cat
from quant_fund.models.monoidal_infty import bench_monoidal_infty
from quant_fund.models.operad_infty import bench_operad_infty


def test_e_n_algebra():
    assert bench_e_n_algebra()["synthetic_e_n_algebra"] == 1.0


def test_operad_infty():
    assert bench_operad_infty()["synthetic_operad_infty"] == 1.0


def test_monoidal_infty():
    assert bench_monoidal_infty()["synthetic_monoidal_infty"] == 1.0


def test_module_cat():
    assert bench_module_cat()["synthetic_module_cat"] == 1.0


def test_brane_tensor():
    assert bench_brane_tensor()["synthetic_brane_tensor"] == 1.0


def test_delooping():
    assert bench_delooping()["synthetic_delooping"] == 1.0
