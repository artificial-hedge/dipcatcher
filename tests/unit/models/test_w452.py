from quant_fund.models.d_module import bench_d_module
from quant_fund.models.geometric_langlands import bench_geometric_langlands
from quant_fund.models.hecke_eig import bench_hecke_eig
from quant_fund.models.kernel_fun import bench_kernel_fun
from quant_fund.models.opers_g import bench_opers_g
from quant_fund.models.ramified_l import bench_ramified_l


def test_d_module():
    assert bench_d_module()["synthetic_d_module"] == 1.0


def test_geometric_langlands():
    assert bench_geometric_langlands()["synthetic_geometric_langlands"] == 1.0


def test_hecke_eig():
    assert bench_hecke_eig()["synthetic_hecke_eig"] == 1.0


def test_opers_g():
    assert bench_opers_g()["synthetic_opers_g"] == 1.0


def test_ramified_l():
    assert bench_ramified_l()["synthetic_ramified_l"] == 1.0


def test_kernel_fun():
    assert bench_kernel_fun()["synthetic_kernel_fun"] == 1.0
