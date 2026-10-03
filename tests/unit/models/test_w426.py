from quant_fund.models.derived_alg import bench_derived_alg
from quant_fund.models.infinity_cat import bench_infinity_cat
from quant_fund.models.model_category import bench_model_category
from quant_fund.models.quillen_adj import bench_quillen_adj
from quant_fund.models.simplicial_set import bench_simplicial_set
from quant_fund.models.stable_cat import bench_stable_cat


def test_model_category():
    assert bench_model_category()["synthetic_model_category"] == 1.0


def test_quillen_adj():
    assert bench_quillen_adj()["synthetic_quillen_adj"] == 1.0


def test_simplicial_set():
    assert bench_simplicial_set()["synthetic_simplicial_set"] == 1.0


def test_infinity_cat():
    assert bench_infinity_cat()["synthetic_infinity_cat"] == 1.0


def test_derived_alg():
    assert bench_derived_alg()["synthetic_derived_alg"] == 1.0


def test_stable_cat():
    assert bench_stable_cat()["synthetic_stable_cat"] == 1.0
