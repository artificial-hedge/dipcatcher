from quant_fund.models.boards_operad import bench_boards_operad
from quant_fund.models.cyclotomic_e_n import bench_cyclotomic_e_n
from quant_fund.models.e3_algebra import bench_e3_algebra
from quant_fund.models.getzler_jones import bench_getzler_jones
from quant_fund.models.surfaces_operad import bench_surfaces_operad
from quant_fund.models.tadv_hochschild import bench_tadv_hochschild


def test_e3_algebra():
    assert bench_e3_algebra()["synthetic_e3_algebra"] == 1.0


def test_getzler_jones():
    assert bench_getzler_jones()["synthetic_getzler_jones"] == 1.0


def test_tadv_hochschild():
    assert bench_tadv_hochschild()["synthetic_tadv_hochschild"] == 1.0


def test_cyclotomic_e_n():
    assert bench_cyclotomic_e_n()["synthetic_cyclotomic_e_n"] == 1.0


def test_surfaces_operad():
    assert bench_surfaces_operad()["synthetic_surfaces_operad"] == 1.0


def test_boards_operad():
    assert bench_boards_operad()["synthetic_boards_operad"] == 1.0
