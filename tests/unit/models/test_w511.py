from quant_fund.models.bun_g import bench_bun_g
from quant_fund.models.fs_diamond import bench_fs_diamond
from quant_fund.models.geometric_satake import bench_geometric_satake
from quant_fund.models.hecke_stack import bench_hecke_stack
from quant_fund.models.v_sheaf import bench_v_sheaf
from quant_fund.models.y_diamond import bench_y_diamond


def test_fs_diamond():
    assert bench_fs_diamond()["synthetic_fs_diamond"] == 1.0


def test_geometric_satake():
    assert bench_geometric_satake()["synthetic_geometric_satake"] == 1.0


def test_v_sheaf():
    assert bench_v_sheaf()["synthetic_v_sheaf"] == 1.0


def test_bun_g():
    assert bench_bun_g()["synthetic_bun_g"] == 1.0


def test_hecke_stack():
    assert bench_hecke_stack()["synthetic_hecke_stack"] == 1.0


def test_y_diamond():
    assert bench_y_diamond()["synthetic_y_diamond"] == 1.0
