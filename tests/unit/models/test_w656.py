from quant_fund.models.admissible_cat import bench_admissible_cat
from quant_fund.models.cartesian_cat2 import bench_cartesian_cat2
from quant_fund.models.cocomplete_cat import bench_cocomplete_cat
from quant_fund.models.definable_cat import bench_definable_cat
from quant_fund.models.essentially_small import bench_essentially_small
from quant_fund.models.finitely_accessible import bench_finitely_accessible


def test_essentially_small():
    assert bench_essentially_small()["synthetic_essentially_small"] == 1.0


def test_finitely_accessible():
    assert bench_finitely_accessible()["synthetic_finitely_accessible"] == 1.0


def test_admissible_cat():
    assert bench_admissible_cat()["synthetic_admissible_cat"] == 1.0


def test_definable_cat():
    assert bench_definable_cat()["synthetic_definable_cat"] == 1.0


def test_cocomplete_cat():
    assert bench_cocomplete_cat()["synthetic_cocomplete_cat"] == 1.0


def test_cartesian_cat2():
    assert bench_cartesian_cat2()["synthetic_cartesian_cat2"] == 1.0
