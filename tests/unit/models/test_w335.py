from quant_fund.models.adjoint_check import bench_adjoint_check
from quant_fund.models.cat_colimit import bench_cat_colimit
from quant_fund.models.exponential_obj import bench_exponential_obj
from quant_fund.models.fin_limit import bench_fin_limit
from quant_fund.models.subobject_classifier import bench_subobject_classifier
from quant_fund.models.yoneda_embed import bench_yoneda_embed


def test_fin_limit():
    assert bench_fin_limit()["synthetic_fin_limit"] == 1.0


def test_subobject_classifier():
    assert bench_subobject_classifier()["synthetic_subobject_classifier"] == 1.0


def test_exponential_obj():
    assert bench_exponential_obj()["synthetic_exponential_obj"] == 1.0


def test_yoneda_embed():
    assert bench_yoneda_embed()["synthetic_yoneda_embed"] == 1.0


def test_adjoint_check():
    assert bench_adjoint_check()["synthetic_adjoint_check"] == 1.0


def test_cat_colimit():
    assert bench_cat_colimit()["synthetic_cat_colimit"] == 1.0
