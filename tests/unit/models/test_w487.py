from quant_fund.models.comma_cat import bench_comma_cat
from quant_fund.models.compact_obj import bench_compact_obj
from quant_fund.models.dualizable_cat import bench_dualizable_cat
from quant_fund.models.endo_prof import bench_endo_prof
from quant_fund.models.exact_cat import bench_exact_cat
from quant_fund.models.prestack import bench_prestack


def test_compact_obj():
    assert bench_compact_obj()["synthetic_compact_obj"] == 1.0


def test_dualizable_cat():
    assert bench_dualizable_cat()["synthetic_dualizable_cat"] == 1.0


def test_comma_cat():
    assert bench_comma_cat()["synthetic_comma_cat"] == 1.0


def test_prestack():
    assert bench_prestack()["synthetic_prestack"] == 1.0


def test_endo_prof():
    assert bench_endo_prof()["synthetic_endo_prof"] == 1.0


def test_exact_cat():
    assert bench_exact_cat()["synthetic_exact_cat"] == 1.0
