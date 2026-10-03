from quant_fund.models.adjoint_op import bench_adjoint_op
from quant_fund.models.compact_resolvent import bench_compact_resolvent
from quant_fund.models.hahn_banach import bench_hahn_banach
from quant_fund.models.projection_thm import bench_projection_thm
from quant_fund.models.riesz_repr import bench_riesz_repr
from quant_fund.models.selfadjoint_spectrum import bench_selfadjoint_spectrum


def test_hahn_banach():
    assert bench_hahn_banach()["synthetic_hahn_banach"] == 1.0


def test_riesz_repr():
    assert bench_riesz_repr()["synthetic_riesz_repr"] == 1.0


def test_adjoint_op():
    assert bench_adjoint_op()["synthetic_adjoint_op"] == 1.0


def test_selfadjoint_spectrum():
    assert bench_selfadjoint_spectrum()["synthetic_selfadjoint_spectrum"] == 1.0


def test_compact_resolvent():
    assert bench_compact_resolvent()["synthetic_compact_resolvent"] == 1.0


def test_projection_thm():
    assert bench_projection_thm()["synthetic_projection_thm"] == 1.0
