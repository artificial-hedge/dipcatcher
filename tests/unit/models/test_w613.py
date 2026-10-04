from quant_fund.models.neron_smooth import bench_neron_smooth
from quant_fund.models.perfect_witt import bench_perfect_witt
from quant_fund.models.semistable_reduction import (
    bench_semistable_reduction,
)
from quant_fund.models.verschiebung_witt import (
    bench_verschiebung_witt,
)
from quant_fund.models.witt_teich import bench_witt_teich
from quant_fund.models.witt_vector import bench_witt_vector


def test_witt_vector():
    assert bench_witt_vector()["synthetic_witt_vector"] == 1.0


def test_witt_teich():
    assert bench_witt_teich()["synthetic_witt_teich"] == 1.0


def test_verschiebung_witt():
    assert bench_verschiebung_witt()["synthetic_verschiebung_witt"] == 1.0


def test_perfect_witt():
    assert bench_perfect_witt()["synthetic_perfect_witt"] == 1.0


def test_neron_smooth():
    assert bench_neron_smooth()["synthetic_neron_smooth"] == 1.0


def test_semistable_reduction():
    assert bench_semistable_reduction()["synthetic_semistable_reduction"] == 1.0
