from quant_fund.models.hodge_decomp import bench_hodge_decomp
from quant_fund.models.l2_hodge import bench_l2_hodge
from quant_fund.models.limit_mhs import bench_limit_mhs
from quant_fund.models.mixed_hodge import bench_mixed_hodge
from quant_fund.models.period_map import bench_period_map
from quant_fund.models.vhs_polarized import bench_vhs_polarized


def test_hodge_decomp():
    assert bench_hodge_decomp()["synthetic_hodge_decomp"] == 1.0


def test_l2_hodge():
    assert bench_l2_hodge()["synthetic_l2_hodge"] == 1.0


def test_mixed_hodge():
    assert bench_mixed_hodge()["synthetic_mixed_hodge"] == 1.0


def test_period_map():
    assert bench_period_map()["synthetic_period_map"] == 1.0


def test_vhs_polarized():
    assert bench_vhs_polarized()["synthetic_vhs_polarized"] == 1.0


def test_limit_mhs():
    assert bench_limit_mhs()["synthetic_limit_mhs"] == 1.0
