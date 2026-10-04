"""Wave-943 matrix-analysis canon tests."""

from __future__ import annotations

from quant_fund.models.gershgorin_disc import bench_gershgorin_disc
from quant_fund.models.kadison_ineq import bench_kadison_ineq
from quant_fund.models.loewner_matrix import bench_loewner_matrix
from quant_fund.models.operator_convex import bench_operator_convex
from quant_fund.models.ostrowski_bound import bench_ostrowski_bound
from quant_fund.models.wielandt_ineq import bench_wielandt_ineq


def test_loewner_matrix():
    assert bench_loewner_matrix()["synthetic_loewner_matrix"] == 1.0


def test_operator_convex():
    assert bench_operator_convex()["synthetic_operator_convex"] == 1.0


def test_kadison_ineq():
    assert bench_kadison_ineq()["synthetic_kadison_ineq"] == 1.0


def test_wielandt_ineq():
    assert bench_wielandt_ineq()["synthetic_wielandt_ineq"] == 1.0


def test_ostrowski_bound():
    assert bench_ostrowski_bound()["synthetic_ostrowski_bound"] == 1.0


def test_gershgorin_disc():
    assert bench_gershgorin_disc()["synthetic_gershgorin_disc"] == 1.0
