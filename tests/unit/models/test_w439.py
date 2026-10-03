from quant_fund.models.formal_group import bench_formal_group
from quant_fund.models.formal_module import bench_formal_module
from quant_fund.models.height_strata import bench_height_strata
from quant_fund.models.lazard_ring import bench_lazard_ring
from quant_fund.models.lubin_tate import bench_lubin_tate
from quant_fund.models.morava_k import bench_morava_k


def test_formal_group():
    assert bench_formal_group()["synthetic_formal_group"] == 1.0


def test_lazard_ring():
    assert bench_lazard_ring()["synthetic_lazard_ring"] == 1.0


def test_formal_module():
    assert bench_formal_module()["synthetic_formal_module"] == 1.0


def test_height_strata():
    assert bench_height_strata()["synthetic_height_strata"] == 1.0


def test_lubin_tate():
    assert bench_lubin_tate()["synthetic_lubin_tate"] == 1.0


def test_morava_k():
    assert bench_morava_k()["synthetic_morava_k"] == 1.0
