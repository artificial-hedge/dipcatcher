from quant_fund.models.dunn_additivity import (
    bench_dunn_additivity,
)
from quant_fund.models.e2_algebra import bench_e2_algebra
from quant_fund.models.khovanov_2 import bench_khovanov_2
from quant_fund.models.mckay_correspond import (
    bench_mckay_correspond,
)
from quant_fund.models.swiss_cheese2 import bench_swiss_cheese2
from quant_fund.models.tensor_factorization import (
    bench_tensor_factorization,
)


def test_e2_algebra():
    assert bench_e2_algebra()["synthetic_e2_algebra"] == 1.0


def test_dunn_additivity():
    assert bench_dunn_additivity()["synthetic_dunn_additivity"] == 1.0


def test_tensor_factorization():
    assert bench_tensor_factorization()["synthetic_tensor_factorization"] == 1.0


def test_swiss_cheese2():
    assert bench_swiss_cheese2()["synthetic_swiss_cheese2"] == 1.0


def test_mckay_correspond():
    assert bench_mckay_correspond()["synthetic_mckay_correspond"] == 1.0


def test_khovanov_2():
    assert bench_khovanov_2()["synthetic_khovanov_2"] == 1.0
