from quant_fund.models.faure_seq import (
    bench_faure_seq,
)
from quant_fund.models.gauss_hermite import (
    bench_gauss_hermite,
)
from quant_fund.models.gauss_laguerre import (
    bench_gauss_laguerre,
)
from quant_fund.models.hiot_decomp import (
    bench_hiot_decomp,
)
from quant_fund.models.importance_mc import (
    bench_importance_mc,
)
from quant_fund.models.tensor_train import (
    bench_tensor_train,
)


def test_faure_seq():
    assert bench_faure_seq()["synthetic_faure_seq"] == 1.0


def test_importance_mc():
    assert bench_importance_mc()["synthetic_importance_mc"] == 1.0


def test_gauss_hermite():
    assert bench_gauss_hermite()["synthetic_gauss_hermite"] == 1.0


def test_gauss_laguerre():
    assert bench_gauss_laguerre()["synthetic_gauss_laguerre"] == 1.0


def test_tensor_train():
    assert bench_tensor_train()["synthetic_tensor_train"] == 1.0


def test_hiot_decomp():
    assert bench_hiot_decomp()["synthetic_hiot_decomp"] == 1.0
