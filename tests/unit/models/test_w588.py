from quant_fund.models.ehp_sequence import bench_ehp_sequence
from quant_fund.models.freudenthal_susp import (
    bench_freudenthal_susp,
)
from quant_fund.models.james_period import bench_james_period
from quant_fund.models.moore_space import bench_moore_space
from quant_fund.models.unstable_adams import bench_unstable_adams
from quant_fund.models.whitehead_prod import bench_whitehead_prod


def test_ehp_sequence():
    assert bench_ehp_sequence()["synthetic_ehp_sequence"] == 1.0


def test_james_period():
    assert bench_james_period()["synthetic_james_period"] == 1.0


def test_whitehead_prod():
    assert bench_whitehead_prod()["synthetic_whitehead_prod"] == 1.0


def test_freudenthal_susp():
    assert bench_freudenthal_susp()["synthetic_freudenthal_susp"] == 1.0


def test_moore_space():
    assert bench_moore_space()["synthetic_moore_space"] == 1.0


def test_unstable_adams():
    assert bench_unstable_adams()["synthetic_unstable_adams"] == 1.0
