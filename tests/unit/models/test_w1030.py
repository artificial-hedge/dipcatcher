"""Wave-1030 electrical-engineering canon tests."""

from __future__ import annotations

from quant_fund.models.circuit_analysis import bench_circuit_analysis
from quant_fund.models.control_systems import bench_control_systems
from quant_fund.models.electromagnetics import bench_electromagnetics
from quant_fund.models.power_systems import bench_power_systems
from quant_fund.models.semiconductor import bench_semiconductor
from quant_fund.models.signal_processing2 import bench_signal_processing2


def test_circuit_analysis():
    assert bench_circuit_analysis()["synthetic_circuit_analysis"] == 1.0


def test_power_systems():
    assert bench_power_systems()["synthetic_power_systems"] == 1.0


def test_control_systems():
    assert bench_control_systems()["synthetic_control_systems"] == 1.0


def test_signal_processing2():
    assert bench_signal_processing2()["synthetic_signal_processing2"] == 1.0


def test_electromagnetics():
    assert bench_electromagnetics()["synthetic_electromagnetics"] == 1.0


def test_semiconductor():
    assert bench_semiconductor()["synthetic_semiconductor"] == 1.0
