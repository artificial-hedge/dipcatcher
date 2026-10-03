from quant_fund.models.busy_beaver import bench_busy_beaver
from quant_fund.models.compactness_lite import bench_compactness_lite
from quant_fund.models.pr_functions import bench_pr_functions
from quant_fund.models.ramsey_theory import bench_ramsey_theory
from quant_fund.models.turing_degrees import bench_turing_degrees
from quant_fund.models.ultraproduct import bench_ultraproduct


def test_pr_functions():
    assert bench_pr_functions()["synthetic_pr_functions"] == 1.0


def test_turing_degrees():
    assert bench_turing_degrees()["synthetic_turing_degrees"] == 1.0


def test_busy_beaver():
    assert bench_busy_beaver()["synthetic_busy_beaver"] == 1.0


def test_ultraproduct():
    assert bench_ultraproduct()["synthetic_ultraproduct"] == 1.0


def test_ramsey_theory():
    assert bench_ramsey_theory()["synthetic_ramsey_theory"] == 1.0


def test_compactness_lite():
    assert bench_compactness_lite()["synthetic_compactness_lite"] == 1.0
