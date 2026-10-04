from quant_fund.models.cusp_form import bench_cusp_form
from quant_fund.models.dedekind_eta import bench_dedekind_eta
from quant_fund.models.eisenstein_srs2 import bench_eisenstein_srs2
from quant_fund.models.hecke_op2 import bench_hecke_op2
from quant_fund.models.modular_form import bench_modular_form
from quant_fund.models.theta_func import bench_theta_func


def test_modular_form():
    assert bench_modular_form()["synthetic_modular_form"] == 1.0


def test_hecke_op2():
    assert bench_hecke_op2()["synthetic_hecke_op2"] == 1.0


def test_eisenstein_srs2():
    assert bench_eisenstein_srs2()["synthetic_eisenstein_srs2"] == 1.0


def test_cusp_form():
    assert bench_cusp_form()["synthetic_cusp_form"] == 1.0


def test_theta_func():
    assert bench_theta_func()["synthetic_theta_func"] == 1.0


def test_dedekind_eta():
    assert bench_dedekind_eta()["synthetic_dedekind_eta"] == 1.0
