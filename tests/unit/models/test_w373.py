from quant_fund.models.bfgs_wolfe import bench_bfgs_wolfe
from quant_fund.models.bundle_method import bench_bundle_method
from quant_fund.models.frank_wolfe2 import bench_frank_wolfe2
from quant_fund.models.ip_qp import bench_ip_qp
from quant_fund.models.sqp import bench_sqp
from quant_fund.models.trust_region import bench_trust_region


def test_bundle_method():
    assert bench_bundle_method()["synthetic_bundle_method"] == 1.0


def test_sqp():
    assert bench_sqp()["synthetic_sqp"] == 1.0


def test_ip_qp():
    assert bench_ip_qp()["synthetic_ip_qp"] == 1.0


def test_trust_region():
    assert bench_trust_region()["synthetic_trust_region"] == 1.0


def test_frank_wolfe2():
    assert bench_frank_wolfe2()["synthetic_frank_wolfe2"] == 1.0


def test_bfgs_wolfe():
    assert bench_bfgs_wolfe()["synthetic_bfgs_wolfe"] == 1.0
