"""Wave-962 von-Neumann-algebra canon tests."""

from __future__ import annotations

from quant_fund.models.double_commutant import bench_double_commutant
from quant_fund.models.jones_index import bench_jones_index
from quant_fund.models.normal_state import bench_normal_state
from quant_fund.models.predual_space import bench_predual_space
from quant_fund.models.tomita_takesaki import bench_tomita_takesaki
from quant_fund.models.von_neumann_alg import bench_von_neumann_alg


def test_von_neumann_alg():
    assert bench_von_neumann_alg()["synthetic_von_neumann_alg"] == 1.0


def test_double_commutant():
    assert bench_double_commutant()["synthetic_double_commutant"] == 1.0


def test_predual_space():
    assert bench_predual_space()["synthetic_predual_space"] == 1.0


def test_normal_state():
    assert bench_normal_state()["synthetic_normal_state"] == 1.0


def test_tomita_takesaki():
    assert bench_tomita_takesaki()["synthetic_tomita_takesaki"] == 1.0


def test_jones_index():
    assert bench_jones_index()["synthetic_jones_index"] == 1.0
