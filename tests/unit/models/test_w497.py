from quant_fund.models.fano_mori import bench_fano_mori
from quant_fund.models.flip_cone import bench_flip_cone
from quant_fund.models.klt_pair import bench_klt_pair
from quant_fund.models.minimal_model import bench_minimal_model
from quant_fund.models.mmp_algorithm import bench_mmp_algorithm
from quant_fund.models.toric_flip import bench_toric_flip


def test_minimal_model():
    assert bench_minimal_model()["synthetic_minimal_model"] == 1.0


def test_klt_pair():
    assert bench_klt_pair()["synthetic_klt_pair"] == 1.0


def test_flip_cone():
    assert bench_flip_cone()["synthetic_flip_cone"] == 1.0


def test_fano_mori():
    assert bench_fano_mori()["synthetic_fano_mori"] == 1.0


def test_mmp_algorithm():
    assert bench_mmp_algorithm()["synthetic_mmp_algorithm"] == 1.0


def test_toric_flip():
    assert bench_toric_flip()["synthetic_toric_flip"] == 1.0
