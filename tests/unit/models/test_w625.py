from quant_fund.models.complexity_spectrum import (
    bench_complexity_spectrum,
)
from quant_fund.models.simplicial_htpy import (
    bench_simplicial_htpy,
)
from quant_fund.models.small_spec import bench_small_spec
from quant_fund.models.spectrum_type import (
    bench_spectrum_type,
)
from quant_fund.models.stable_cohomology2 import (
    bench_stable_cohomology2,
)
from quant_fund.models.woodward_op import (
    bench_woodward_op,
)


def test_stable_cohomology2():
    assert bench_stable_cohomology2()["synthetic_stable_cohomology2"] == 1.0


def test_woodward_op():
    assert bench_woodward_op()["synthetic_woodward_op"] == 1.0


def test_spectrum_type():
    assert bench_spectrum_type()["synthetic_spectrum_type"] == 1.0


def test_complexity_spectrum():
    assert bench_complexity_spectrum()["synthetic_complexity_spectrum"] == 1.0


def test_small_spec():
    assert bench_small_spec()["synthetic_small_spec"] == 1.0


def test_simplicial_htpy():
    assert bench_simplicial_htpy()["synthetic_simplicial_htpy"] == 1.0
