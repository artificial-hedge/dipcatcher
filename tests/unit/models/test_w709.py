from quant_fund.models.cat_ab2 import bench_cat_ab2
from quant_fund.models.cat_ab_loc import bench_cat_ab_loc
from quant_fund.models.cat_exact3 import bench_cat_exact3
from quant_fund.models.cat_freyd import bench_cat_freyd
from quant_fund.models.cat_pro_object2 import (
    bench_cat_pro_object2,
)
from quant_fund.models.cat_univariant2 import (
    bench_cat_univariant2,
)


def test_cat_univariant2():
    assert bench_cat_univariant2()["synthetic_cat_univariant2"] == 1.0


def test_cat_ab2():
    assert bench_cat_ab2()["synthetic_cat_ab2"] == 1.0


def test_cat_exact3():
    assert bench_cat_exact3()["synthetic_cat_exact3"] == 1.0


def test_cat_freyd():
    assert bench_cat_freyd()["synthetic_cat_freyd"] == 1.0


def test_cat_ab_loc():
    assert bench_cat_ab_loc()["synthetic_cat_ab_loc"] == 1.0


def test_cat_pro_object2():
    assert bench_cat_pro_object2()["synthetic_cat_pro_object2"] == 1.0
