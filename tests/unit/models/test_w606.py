from quant_fund.models.connective_k import bench_connective_k
from quant_fund.models.higher_k import bench_higher_k
from quant_fund.models.k_spectrum import bench_k_spectrum
from quant_fund.models.karoubi_k import bench_karoubi_k
from quant_fund.models.nil_k import bench_nil_k
from quant_fund.models.pedersen_weibel import (
    bench_pedersen_weibel,
)


def test_connective_k():
    assert bench_connective_k()["synthetic_connective_k"] == 1.0


def test_higher_k():
    assert bench_higher_k()["synthetic_higher_k"] == 1.0


def test_k_spectrum():
    assert bench_k_spectrum()["synthetic_k_spectrum"] == 1.0


def test_nil_k():
    assert bench_nil_k()["synthetic_nil_k"] == 1.0


def test_karoubi_k():
    assert bench_karoubi_k()["synthetic_karoubi_k"] == 1.0


def test_pedersen_weibel():
    assert bench_pedersen_weibel()["synthetic_pedersen_weibel"] == 1.0
