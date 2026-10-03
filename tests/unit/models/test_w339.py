from quant_fund.models.field_ext import bench_field_ext
from quant_fund.models.galois_group import bench_galois_group
from quant_fund.models.lie_bracket import bench_lie_bracket
from quant_fund.models.rep_theory import bench_rep_theory
from quant_fund.models.root_system import bench_root_system
from quant_fund.models.splitting_field import bench_splitting_field


def test_field_ext():
    assert bench_field_ext()["synthetic_field_ext"] == 1.0


def test_galois_group():
    assert bench_galois_group()["synthetic_galois_group"] == 1.0


def test_splitting_field():
    assert bench_splitting_field()["synthetic_splitting_field"] == 1.0


def test_lie_bracket():
    assert bench_lie_bracket()["synthetic_lie_bracket"] == 1.0


def test_rep_theory():
    assert bench_rep_theory()["synthetic_rep_theory"] == 1.0


def test_root_system():
    assert bench_root_system()["synthetic_root_system"] == 1.0
