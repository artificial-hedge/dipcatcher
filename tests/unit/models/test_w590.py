from quant_fund.models.bloch_k import bench_bloch_k
from quant_fund.models.gersten_ss import bench_gersten_ss
from quant_fund.models.loday_k import bench_loday_k
from quant_fund.models.quillen_plus import bench_quillen_plus
from quant_fund.models.suslin_k import bench_suslin_k
from quant_fund.models.volodin_k import bench_volodin_k


def test_quillen_plus():
    assert bench_quillen_plus()["synthetic_quillen_plus"] == 1.0


def test_gersten_ss():
    assert bench_gersten_ss()["synthetic_gersten_ss"] == 1.0


def test_loday_k():
    assert bench_loday_k()["synthetic_loday_k"] == 1.0


def test_volodin_k():
    assert bench_volodin_k()["synthetic_volodin_k"] == 1.0


def test_suslin_k():
    assert bench_suslin_k()["synthetic_suslin_k"] == 1.0


def test_bloch_k():
    assert bench_bloch_k()["synthetic_bloch_k"] == 1.0
