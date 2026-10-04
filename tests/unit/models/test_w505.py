from quant_fund.models.cobordism_grp import bench_cobordism_grp
from quant_fund.models.complex_cob import bench_complex_cob
from quant_fund.models.framed_cob import bench_framed_cob
from quant_fund.models.oriented_cob import bench_oriented_cob
from quant_fund.models.thom_cob import bench_thom_cob
from quant_fund.models.unoriented_cob import bench_unoriented_cob


def test_cobordism_grp():
    assert bench_cobordism_grp()["synthetic_cobordism_grp"] == 1.0


def test_oriented_cob():
    assert bench_oriented_cob()["synthetic_oriented_cob"] == 1.0


def test_unoriented_cob():
    assert bench_unoriented_cob()["synthetic_unoriented_cob"] == 1.0


def test_complex_cob():
    assert bench_complex_cob()["synthetic_complex_cob"] == 1.0


def test_framed_cob():
    assert bench_framed_cob()["synthetic_framed_cob"] == 1.0


def test_thom_cob():
    assert bench_thom_cob()["synthetic_thom_cob"] == 1.0
