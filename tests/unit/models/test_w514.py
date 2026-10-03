from quant_fund.models.beilinson_reg import bench_beilinson_reg
from quant_fund.models.cheeger_simons import bench_cheeger_simons
from quant_fund.models.deligne_cohom import bench_deligne_cohom
from quant_fund.models.diff_cohom import bench_diff_cohom
from quant_fund.models.flat_bundle import bench_flat_bundle
from quant_fund.models.secondary_inv import bench_secondary_inv


def test_diff_cohom():
    assert bench_diff_cohom()["synthetic_diff_cohom"] == 1.0


def test_cheeger_simons():
    assert bench_cheeger_simons()["synthetic_cheeger_simons"] == 1.0


def test_deligne_cohom():
    assert bench_deligne_cohom()["synthetic_deligne_cohom"] == 1.0


def test_flat_bundle():
    assert bench_flat_bundle()["synthetic_flat_bundle"] == 1.0


def test_beilinson_reg():
    assert bench_beilinson_reg()["synthetic_beilinson_reg"] == 1.0


def test_secondary_inv():
    assert bench_secondary_inv()["synthetic_secondary_inv"] == 1.0
