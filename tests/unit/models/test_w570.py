from quant_fund.models.bogomolov_ineq import bench_bogomolov_ineq
from quant_fund.models.boundedness_moduli import (
    bench_boundedness_moduli,
)
from quant_fund.models.hodge_index import bench_hodge_index
from quant_fund.models.kodaira_vanishing import (
    bench_kodaira_vanishing,
)
from quant_fund.models.kollar_mori import bench_kollar_mori
from quant_fund.models.stability_sheaf import bench_stability_sheaf


def test_hodge_index():
    assert bench_hodge_index()["synthetic_hodge_index"] == 1.0


def test_kodaira_vanishing():
    assert bench_kodaira_vanishing()["synthetic_kodaira_vanishing"] == 1.0


def test_kollar_mori():
    assert bench_kollar_mori()["synthetic_kollar_mori"] == 1.0


def test_boundedness_moduli():
    assert bench_boundedness_moduli()["synthetic_boundedness_moduli"] == 1.0


def test_stability_sheaf():
    assert bench_stability_sheaf()["synthetic_stability_sheaf"] == 1.0


def test_bogomolov_ineq():
    assert bench_bogomolov_ineq()["synthetic_bogomolov_ineq"] == 1.0
