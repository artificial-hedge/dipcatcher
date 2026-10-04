from quant_fund.models.gue_statistics import bench_gue_statistics
from quant_fund.models.keating_snaith import bench_keating_snaith
from quant_fund.models.montgomery_pair import (
    bench_montgomery_pair,
)
from quant_fund.models.rudnick_sarnak import bench_rudnick_sarnak
from quant_fund.models.selberg_trace2 import bench_selberg_trace2
from quant_fund.models.zero_spacing import bench_zero_spacing


def test_selberg_trace2():
    assert bench_selberg_trace2()["synthetic_selberg_trace2"] == 1.0


def test_zero_spacing():
    assert bench_zero_spacing()["synthetic_zero_spacing"] == 1.0


def test_montgomery_pair():
    assert bench_montgomery_pair()["synthetic_montgomery_pair"] == 1.0


def test_gue_statistics():
    assert bench_gue_statistics()["synthetic_gue_statistics"] == 1.0


def test_keating_snaith():
    assert bench_keating_snaith()["synthetic_keating_snaith"] == 1.0


def test_rudnick_sarnak():
    assert bench_rudnick_sarnak()["synthetic_rudnick_sarnak"] == 1.0
