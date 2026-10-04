from quant_fund.models.free_convolution import (
    bench_free_convolution,
)
from quant_fund.models.free_prob import bench_free_prob
from quant_fund.models.operator_valued import (
    bench_operator_valued,
)
from quant_fund.models.r_transform import bench_r_transform
from quant_fund.models.s_transform import bench_s_transform
from quant_fund.models.voiculescu_thm import (
    bench_voiculescu_thm,
)


def test_free_prob():
    assert bench_free_prob()["synthetic_free_prob"] == 1.0


def test_r_transform():
    assert bench_r_transform()["synthetic_r_transform"] == 1.0


def test_s_transform():
    assert bench_s_transform()["synthetic_s_transform"] == 1.0


def test_free_convolution():
    assert bench_free_convolution()["synthetic_free_convolution"] == 1.0


def test_voiculescu_thm():
    assert bench_voiculescu_thm()["synthetic_voiculescu_thm"] == 1.0


def test_operator_valued():
    assert bench_operator_valued()["synthetic_operator_valued"] == 1.0
