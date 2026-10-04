from quant_fund.models.complicial import bench_complicial
from quant_fund.models.globular_model import bench_globular_model
from quant_fund.models.opetopic import bench_opetopic
from quant_fund.models.theta_space import bench_theta_space
from quant_fund.models.verity_gray import bench_verity_gray
from quant_fund.models.weak_infty import bench_weak_infty


def test_globular_model():
    assert bench_globular_model()["synthetic_globular_model"] == 1.0


def test_opetopic():
    assert bench_opetopic()["synthetic_opetopic"] == 1.0


def test_theta_space():
    assert bench_theta_space()["synthetic_theta_space"] == 1.0


def test_complicial():
    assert bench_complicial()["synthetic_complicial"] == 1.0


def test_verity_gray():
    assert bench_verity_gray()["synthetic_verity_gray"] == 1.0


def test_weak_infty():
    assert bench_weak_infty()["synthetic_weak_infty"] == 1.0
