from quant_fund.models.brace_operad import bench_brace_operad
from quant_fund.models.little_intervals import bench_little_intervals
from quant_fund.models.operad_algt import bench_operad_algt
from quant_fund.models.operad_homology import bench_operad_homology
from quant_fund.models.props_toy import bench_props_toy
from quant_fund.models.swiss_cheese import bench_swiss_cheese


def test_operad_algt():
    assert bench_operad_algt()["synthetic_operad_algt"] == 1.0


def test_brace_operad():
    assert bench_brace_operad()["synthetic_brace_operad"] == 1.0


def test_swiss_cheese():
    assert bench_swiss_cheese()["synthetic_swiss_cheese"] == 1.0


def test_little_intervals():
    assert bench_little_intervals()["synthetic_little_intervals"] == 1.0


def test_operad_homology():
    assert bench_operad_homology()["synthetic_operad_homology"] == 1.0


def test_props_toy():
    assert bench_props_toy()["synthetic_props_toy"] == 1.0
