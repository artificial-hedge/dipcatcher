from quant_fund.models.derived_abelian2 import (
    bench_derived_abelian2,
)
from quant_fund.models.derived_cover import bench_derived_cover
from quant_fund.models.derived_geometry7 import (
    bench_derived_geometry7,
)
from quant_fund.models.derived_morph import bench_derived_morph
from quant_fund.models.derived_stack3 import bench_derived_stack3
from quant_fund.models.derived_topos import bench_derived_topos


def test_derived_geometry7():
    assert bench_derived_geometry7()["synthetic_derived_geometry7"] == 1.0


def test_derived_abelian2():
    assert bench_derived_abelian2()["synthetic_derived_abelian2"] == 1.0


def test_derived_stack3():
    assert bench_derived_stack3()["synthetic_derived_stack3"] == 1.0


def test_derived_morph():
    assert bench_derived_morph()["synthetic_derived_morph"] == 1.0


def test_derived_cover():
    assert bench_derived_cover()["synthetic_derived_cover"] == 1.0


def test_derived_topos():
    assert bench_derived_topos()["synthetic_derived_topos"] == 1.0
