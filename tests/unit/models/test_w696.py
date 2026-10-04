from quant_fund.models.cat_image import bench_cat_image
from quant_fund.models.cat_index import bench_cat_index
from quant_fund.models.cat_kernel import bench_cat_kernel
from quant_fund.models.cat_monotone import (
    bench_cat_monotone,
)
from quant_fund.models.cat_pullback import (
    bench_cat_pullback,
)
from quant_fund.models.cat_rank import bench_cat_rank


def test_cat_rank():
    assert bench_cat_rank()["synthetic_cat_rank"] == 1.0


def test_cat_index():
    assert bench_cat_index()["synthetic_cat_index"] == 1.0


def test_cat_monotone():
    assert bench_cat_monotone()["synthetic_cat_monotone"] == 1.0


def test_cat_kernel():
    assert bench_cat_kernel()["synthetic_cat_kernel"] == 1.0


def test_cat_image():
    assert bench_cat_image()["synthetic_cat_image"] == 1.0


def test_cat_pullback():
    assert bench_cat_pullback()["synthetic_cat_pullback"] == 1.0
