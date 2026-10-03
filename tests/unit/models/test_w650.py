from quant_fund.models.anick_htpy import bench_anick_htpy
from quant_fund.models.bousfield_htpy import bench_bousfield_htpy
from quant_fund.models.dror_htpy import bench_dror_htpy
from quant_fund.models.kane_htpy import bench_kane_htpy
from quant_fund.models.moore_htpy import bench_moore_htpy
from quant_fund.models.neisendorfer_htpy import bench_neisendorfer_htpy


def test_bousfield_htpy():
    assert bench_bousfield_htpy()["synthetic_bousfield_htpy"] == 1.0


def test_dror_htpy():
    assert bench_dror_htpy()["synthetic_dror_htpy"] == 1.0


def test_kane_htpy():
    assert bench_kane_htpy()["synthetic_kane_htpy"] == 1.0


def test_moore_htpy():
    assert bench_moore_htpy()["synthetic_moore_htpy"] == 1.0


def test_neisendorfer_htpy():
    assert bench_neisendorfer_htpy()["synthetic_neisendorfer_htpy"] == 1.0


def test_anick_htpy():
    assert bench_anick_htpy()["synthetic_anick_htpy"] == 1.0
