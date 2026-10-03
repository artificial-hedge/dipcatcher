from quant_fund.models.artinian_alg import (
    bench_artinian_alg,
)
from quant_fund.models.deform_functor2 import (
    bench_deform_functor2,
)
from quant_fund.models.hull_deform import (
    bench_hull_deform,
)
from quant_fund.models.rim_deform import (
    bench_rim_deform,
)
from quant_fund.models.small_ext import bench_small_ext
from quant_fund.models.tangent_def import (
    bench_tangent_def,
)


def test_deform_functor2():
    assert bench_deform_functor2()["synthetic_deform_functor2"] == 1.0


def test_tangent_def():
    assert bench_tangent_def()["synthetic_tangent_def"] == 1.0


def test_rim_deform():
    assert bench_rim_deform()["synthetic_rim_deform"] == 1.0


def test_small_ext():
    assert bench_small_ext()["synthetic_small_ext"] == 1.0


def test_hull_deform():
    assert bench_hull_deform()["synthetic_hull_deform"] == 1.0


def test_artinian_alg():
    assert bench_artinian_alg()["synthetic_artinian_alg"] == 1.0
