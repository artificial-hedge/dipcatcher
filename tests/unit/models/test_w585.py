from quant_fund.models.canonical_bundle import (
    bench_canonical_bundle,
)
from quant_fund.models.intersection_theory import (
    bench_intersection_theory,
)
from quant_fund.models.line_bundle import bench_line_bundle
from quant_fund.models.macpherson_chern import (
    bench_macpherson_chern,
)
from quant_fund.models.picard_group import bench_picard_group
from quant_fund.models.weil_divisor import bench_weil_divisor


def test_intersection_theory():
    assert bench_intersection_theory()["synthetic_intersection_theory"] == 1.0


def test_macpherson_chern():
    assert bench_macpherson_chern()["synthetic_macpherson_chern"] == 1.0


def test_weil_divisor():
    assert bench_weil_divisor()["synthetic_weil_divisor"] == 1.0


def test_picard_group():
    assert bench_picard_group()["synthetic_picard_group"] == 1.0


def test_line_bundle():
    assert bench_line_bundle()["synthetic_line_bundle"] == 1.0


def test_canonical_bundle():
    assert bench_canonical_bundle()["synthetic_canonical_bundle"] == 1.0
