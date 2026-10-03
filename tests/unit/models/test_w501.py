from quant_fund.models.f_pure import bench_f_pure
from quant_fund.models.f_rational import bench_f_rational
from quant_fund.models.f_regular import bench_f_regular
from quant_fund.models.f_threshold import bench_f_threshold
from quant_fund.models.test_ideal import bench_test_ideal
from quant_fund.models.tight_closure import bench_tight_closure


def test_f_regular():
    assert bench_f_regular()["synthetic_f_regular"] == 1.0


def test_f_rational():
    assert bench_f_rational()["synthetic_f_rational"] == 1.0


def test_f_pure():
    assert bench_f_pure()["synthetic_f_pure"] == 1.0


def test_f_threshold():
    assert bench_f_threshold()["synthetic_f_threshold"] == 1.0


def test_test_ideal():
    assert bench_test_ideal()["synthetic_test_ideal"] == 1.0


def test_tight_closure():
    assert bench_tight_closure()["synthetic_tight_closure"] == 1.0
