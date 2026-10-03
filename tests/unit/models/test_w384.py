from quant_fund.models.aut_group import bench_aut_group
from quant_fund.models.composition_series import bench_composition_series
from quant_fund.models.hall_subgroup import bench_hall_subgroup
from quant_fund.models.permutation_poly import bench_permutation_poly
from quant_fund.models.schur_multiplier import bench_schur_multiplier
from quant_fund.models.transfer_hom import bench_transfer_hom


def test_hall_subgroup():
    assert bench_hall_subgroup()["synthetic_hall_subgroup"] == 1.0


def test_transfer_hom():
    assert bench_transfer_hom()["synthetic_transfer_hom"] == 1.0


def test_schur_multiplier():
    assert bench_schur_multiplier()["synthetic_schur_multiplier"] == 1.0


def test_aut_group():
    assert bench_aut_group()["synthetic_aut_group"] == 1.0


def test_composition_series():
    assert bench_composition_series()["synthetic_composition_series"] == 1.0


def test_permutation_poly():
    assert bench_permutation_poly()["synthetic_permutation_poly"] == 1.0
