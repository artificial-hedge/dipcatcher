from quant_fund.models.artin_symbol import bench_artin_symbol
from quant_fund.models.class_group_toy import bench_class_group_toy
from quant_fund.models.decomposition_group import bench_decomposition_group
from quant_fund.models.discriminant_field import bench_discriminant_field
from quant_fund.models.norm_subring import bench_norm_subring
from quant_fund.models.ramification import bench_ramification


def test_norm_subring():
    assert bench_norm_subring()["synthetic_norm_subring"] == 1.0


def test_discriminant_field():
    assert bench_discriminant_field()["synthetic_discriminant_field"] == 1.0


def test_decomposition_group():
    assert bench_decomposition_group()["synthetic_decomposition_group"] == 1.0


def test_ramification():
    assert bench_ramification()["synthetic_ramification"] == 1.0


def test_artin_symbol():
    assert bench_artin_symbol()["synthetic_artin_symbol"] == 1.0


def test_class_group_toy():
    assert bench_class_group_toy()["synthetic_class_group_toy"] == 1.0
