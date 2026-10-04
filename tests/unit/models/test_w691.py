from quant_fund.models.cat_fusion import bench_cat_fusion
from quant_fund.models.cat_pretopos import bench_cat_pretopos
from quant_fund.models.cat_ribbon import bench_cat_ribbon
from quant_fund.models.cat_semiadd import bench_cat_semiadd
from quant_fund.models.cat_semisimple import (
    bench_cat_semisimple,
)
from quant_fund.models.cat_tannakian2 import (
    bench_cat_tannakian2,
)


def test_cat_pretopos():
    assert bench_cat_pretopos()["synthetic_cat_pretopos"] == 1.0


def test_cat_semisimple():
    assert bench_cat_semisimple()["synthetic_cat_semisimple"] == 1.0


def test_cat_fusion():
    assert bench_cat_fusion()["synthetic_cat_fusion"] == 1.0


def test_cat_tannakian2():
    assert bench_cat_tannakian2()["synthetic_cat_tannakian2"] == 1.0


def test_cat_ribbon():
    assert bench_cat_ribbon()["synthetic_cat_ribbon"] == 1.0


def test_cat_semiadd():
    assert bench_cat_semiadd()["synthetic_cat_semiadd"] == 1.0
