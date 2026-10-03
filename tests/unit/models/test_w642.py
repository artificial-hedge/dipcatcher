from quant_fund.models.bhatt_scholze import (
    bench_bhatt_scholze,
)
from quant_fund.models.derived_prism import (
    bench_derived_prism,
)
from quant_fund.models.prismatic_dieudonne import (
    bench_prismatic_dieudonne,
)
from quant_fund.models.prismatic_f import (
    bench_prismatic_f,
)
from quant_fund.models.q_crystal import bench_q_crystal
from quant_fund.models.q_prism import bench_q_prism


def test_prismatic_f():
    assert bench_prismatic_f()["synthetic_prismatic_f"] == 1.0


def test_bhatt_scholze():
    assert bench_bhatt_scholze()["synthetic_bhatt_scholze"] == 1.0


def test_q_crystal():
    assert bench_q_crystal()["synthetic_q_crystal"] == 1.0


def test_prismatic_dieudonne():
    assert bench_prismatic_dieudonne()["synthetic_prismatic_dieudonne"] == 1.0


def test_q_prism():
    assert bench_q_prism()["synthetic_q_prism"] == 1.0


def test_derived_prism():
    assert bench_derived_prism()["synthetic_derived_prism"] == 1.0
