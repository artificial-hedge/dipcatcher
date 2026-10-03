from quant_fund.models.box_counting import bench_box_counting
from quant_fund.models.frostman import bench_frostman
from quant_fund.models.hausdorff_dim import bench_hausdorff_dim
from quant_fund.models.iterated_function import bench_iterated_function
from quant_fund.models.multifractal_formal import (
    bench_multifractal_formal,
)
from quant_fund.models.self_similar import bench_self_similar


def test_hausdorff_dim():
    assert bench_hausdorff_dim()["synthetic_hausdorff_dim"] == 1.0


def test_box_counting():
    assert bench_box_counting()["synthetic_box_counting"] == 1.0


def test_self_similar():
    assert bench_self_similar()["synthetic_self_similar"] == 1.0


def test_iterated_function():
    assert bench_iterated_function()["synthetic_iterated_function"] == 1.0


def test_frostman():
    assert bench_frostman()["synthetic_frostman"] == 1.0


def test_multifractal_formal():
    assert bench_multifractal_formal()["synthetic_multifractal_formal"] == 1.0
