"""Wave-1004 kinetic-theory canon tests."""

from __future__ import annotations

from quant_fund.models.bgk_model import bench_bgk_model
from quant_fund.models.boltzmann_eq import bench_boltzmann_eq
from quant_fund.models.chapman_enskog import bench_chapman_enskog
from quant_fund.models.h_theorem import bench_h_theorem
from quant_fund.models.landau_damping import bench_landau_damping
from quant_fund.models.vlasov_eq import bench_vlasov_eq


def test_boltzmann_eq():
    assert bench_boltzmann_eq()["synthetic_boltzmann_eq"] == 1.0


def test_vlasov_eq():
    assert bench_vlasov_eq()["synthetic_vlasov_eq"] == 1.0


def test_bgk_model():
    assert bench_bgk_model()["synthetic_bgk_model"] == 1.0


def test_chapman_enskog():
    assert bench_chapman_enskog()["synthetic_chapman_enskog"] == 1.0


def test_h_theorem():
    assert bench_h_theorem()["synthetic_h_theorem"] == 1.0


def test_landau_damping():
    assert bench_landau_damping()["synthetic_landau_damping"] == 1.0
