from quant_fund.models.berrick_k import bench_berrick_k
from quant_fund.models.gersen_suslin import bench_gersen_suslin
from quant_fund.models.gillet_thomason import (
    bench_gillet_thomason,
)
from quant_fund.models.hermitian_quillen import (
    bench_hermitian_quillen,
)
from quant_fund.models.k_theory4 import bench_k_theory4
from quant_fund.models.khomo_k import bench_khomo_k


def test_gillet_thomason():
    assert bench_gillet_thomason()["synthetic_gillet_thomason"] == 1.0


def test_khomo_k():
    assert bench_khomo_k()["synthetic_khomo_k"] == 1.0


def test_k_theory4():
    assert bench_k_theory4()["synthetic_k_theory4"] == 1.0


def test_gersen_suslin():
    assert bench_gersen_suslin()["synthetic_gersen_suslin"] == 1.0


def test_berrick_k():
    assert bench_berrick_k()["synthetic_berrick_k"] == 1.0


def test_hermitian_quillen():
    assert bench_hermitian_quillen()["synthetic_hermitian_quillen"] == 1.0
