from quant_fund.models.accessible_cat2 import bench_accessible_cat2
from quant_fund.models.compactly_generated import bench_compactly_generated
from quant_fund.models.flat_monad import bench_flat_monad
from quant_fund.models.locally_presentable import bench_locally_presentable
from quant_fund.models.presentable_cat2 import bench_presentable_cat2
from quant_fund.models.regular_cat2 import bench_regular_cat2


def test_compactly_generated():
    assert bench_compactly_generated()["synthetic_compactly_generated"] == 1.0


def test_presentable_cat2():
    assert bench_presentable_cat2()["synthetic_presentable_cat2"] == 1.0


def test_accessible_cat2():
    assert bench_accessible_cat2()["synthetic_accessible_cat2"] == 1.0


def test_flat_monad():
    assert bench_flat_monad()["synthetic_flat_monad"] == 1.0


def test_locally_presentable():
    assert bench_locally_presentable()["synthetic_locally_presentable"] == 1.0


def test_regular_cat2():
    assert bench_regular_cat2()["synthetic_regular_cat2"] == 1.0
