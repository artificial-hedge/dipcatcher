from quant_fund.models.analytic_torsion import bench_analytic_torsion
from quant_fund.models.atiyah_singer import bench_atiyah_singer
from quant_fund.models.dirac_op import bench_dirac_op
from quant_fund.models.eta_invariant import bench_eta_invariant
from quant_fund.models.heat_kernel2 import bench_heat_kernel2
from quant_fund.models.signature_op import bench_signature_op


def test_atiyah_singer():
    assert bench_atiyah_singer()["synthetic_atiyah_singer"] == 1.0


def test_dirac_op():
    assert bench_dirac_op()["synthetic_dirac_op"] == 1.0


def test_eta_invariant():
    assert bench_eta_invariant()["synthetic_eta_invariant"] == 1.0


def test_heat_kernel2():
    assert bench_heat_kernel2()["synthetic_heat_kernel2"] == 1.0


def test_signature_op():
    assert bench_signature_op()["synthetic_signature_op"] == 1.0


def test_analytic_torsion():
    assert bench_analytic_torsion()["synthetic_analytic_torsion"] == 1.0
