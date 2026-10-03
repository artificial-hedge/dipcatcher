from quant_fund.models.fundamental_cat import (
    bench_fundamental_cat,
)
from quant_fund.models.grayson_s import bench_grayson_s
from quant_fund.models.karoubi_v2 import (
    bench_karoubi_v2,
)
from quant_fund.models.quillen_ldev import (
    bench_quillen_ldev,
)
from quant_fund.models.seg_street import (
    bench_seg_street,
)
from quant_fund.models.vorst_descent import (
    bench_vorst_descent,
)


def test_grayson_s():
    assert bench_grayson_s()["synthetic_grayson_s"] == 1.0


def test_karoubi_v2():
    assert bench_karoubi_v2()["synthetic_karoubi_v2"] == 1.0


def test_vorst_descent():
    assert bench_vorst_descent()["synthetic_vorst_descent"] == 1.0


def test_quillen_ldev():
    assert bench_quillen_ldev()["synthetic_quillen_ldev"] == 1.0


def test_fundamental_cat():
    assert bench_fundamental_cat()["synthetic_fundamental_cat"] == 1.0


def test_seg_street():
    assert bench_seg_street()["synthetic_seg_street"] == 1.0
