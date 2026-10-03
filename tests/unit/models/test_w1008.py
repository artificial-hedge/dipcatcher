"""Wave-1008 thermodynamics canon tests."""

from __future__ import annotations

from quant_fund.models.carnot_cycle import bench_carnot_cycle
from quant_fund.models.critical_phenomena import bench_critical_phenomena
from quant_fund.models.entropy_production import bench_entropy_production
from quant_fund.models.fluctuation_dissipation import bench_fluctuation_dissipation
from quant_fund.models.maxwell_relations import bench_maxwell_relations
from quant_fund.models.phase_transitions import bench_phase_transitions


def test_carnot_cycle():
    assert bench_carnot_cycle()["synthetic_carnot_cycle"] == 1.0


def test_maxwell_relations():
    assert bench_maxwell_relations()["synthetic_maxwell_relations"] == 1.0


def test_phase_transitions():
    assert bench_phase_transitions()["synthetic_phase_transitions"] == 1.0


def test_critical_phenomena():
    assert bench_critical_phenomena()["synthetic_critical_phenomena"] == 1.0


def test_fluctuation_dissipation():
    assert bench_fluctuation_dissipation()["synthetic_fluctuation_dissipation"] == 1.0


def test_entropy_production():
    assert bench_entropy_production()["synthetic_entropy_production"] == 1.0
