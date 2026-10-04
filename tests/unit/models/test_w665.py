from quant_fund.models.cohen_moore2 import bench_cohen_moore2
from quant_fund.models.homotopy_decomp import (
    bench_homotopy_decomp,
)
from quant_fund.models.kervaire_inv2 import bench_kervaire_inv2
from quant_fund.models.moore_space2 import bench_moore_space2
from quant_fund.models.unstable_vn import bench_unstable_vn
from quant_fund.models.whitehead_product import (
    bench_whitehead_product,
)


def test_cohen_moore2():
    assert bench_cohen_moore2()["synthetic_cohen_moore2"] == 1.0


def test_whitehead_product():
    assert bench_whitehead_product()["synthetic_whitehead_product"] == 1.0


def test_homotopy_decomp():
    assert bench_homotopy_decomp()["synthetic_homotopy_decomp"] == 1.0


def test_kervaire_inv2():
    assert bench_kervaire_inv2()["synthetic_kervaire_inv2"] == 1.0


def test_unstable_vn():
    assert bench_unstable_vn()["synthetic_unstable_vn"] == 1.0


def test_moore_space2():
    assert bench_moore_space2()["synthetic_moore_space2"] == 1.0
