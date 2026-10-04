"""Wave-1012 nuclear/particle-physics canon tests."""

from __future__ import annotations

from quant_fund.models.bcs_theory import bench_bcs_theory
from quant_fund.models.cabibbo_km import bench_cabibbo_km
from quant_fund.models.nuclear_liquid_drop import bench_nuclear_liquid_drop
from quant_fund.models.nuclear_shell_model import bench_nuclear_shell_model
from quant_fund.models.parton_model import bench_parton_model
from quant_fund.models.quark_model import bench_quark_model


def test_bcs_theory():
    assert bench_bcs_theory()["synthetic_bcs_theory"] == 1.0


def test_nuclear_shell_model():
    assert bench_nuclear_shell_model()["synthetic_nuclear_shell_model"] == 1.0


def test_nuclear_liquid_drop():
    assert bench_nuclear_liquid_drop()["synthetic_nuclear_liquid_drop"] == 1.0


def test_quark_model():
    assert bench_quark_model()["synthetic_quark_model"] == 1.0


def test_parton_model():
    assert bench_parton_model()["synthetic_parton_model"] == 1.0


def test_cabibbo_km():
    assert bench_cabibbo_km()["synthetic_cabibbo_km"] == 1.0
