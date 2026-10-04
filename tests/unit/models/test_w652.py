from quant_fund.models.asymptotic_motive import bench_asymptotic_motive
from quant_fund.models.exponential_motive import bench_exponential_motive
from quant_fund.models.log_motive import bench_log_motive
from quant_fund.models.numerical_motive import bench_numerical_motive
from quant_fund.models.sheaf_motive import bench_sheaf_motive
from quant_fund.models.strict_motive import bench_strict_motive


def test_strict_motive():
    assert bench_strict_motive()["synthetic_strict_motive"] == 1.0


def test_sheaf_motive():
    assert bench_sheaf_motive()["synthetic_sheaf_motive"] == 1.0


def test_numerical_motive():
    assert bench_numerical_motive()["synthetic_numerical_motive"] == 1.0


def test_asymptotic_motive():
    assert bench_asymptotic_motive()["synthetic_asymptotic_motive"] == 1.0


def test_exponential_motive():
    assert bench_exponential_motive()["synthetic_exponential_motive"] == 1.0


def test_log_motive():
    assert bench_log_motive()["synthetic_log_motive"] == 1.0
