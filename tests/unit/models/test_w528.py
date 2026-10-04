from quant_fund.models.equilibrium_state import bench_equilibrium_state
from quant_fund.models.lasota_yorke import bench_lasota_yorke
from quant_fund.models.pressure_thm import bench_pressure_thm
from quant_fund.models.ruelle_zeta import bench_ruelle_zeta
from quant_fund.models.thermo_formal import bench_thermo_formal
from quant_fund.models.transfer_op import bench_transfer_op


def test_transfer_op():
    assert bench_transfer_op()["synthetic_transfer_op"] == 1.0


def test_thermo_formal():
    assert bench_thermo_formal()["synthetic_thermo_formal"] == 1.0


def test_pressure_thm():
    assert bench_pressure_thm()["synthetic_pressure_thm"] == 1.0


def test_equilibrium_state():
    assert bench_equilibrium_state()["synthetic_equilibrium_state"] == 1.0


def test_ruelle_zeta():
    assert bench_ruelle_zeta()["synthetic_ruelle_zeta"] == 1.0


def test_lasota_yorke():
    assert bench_lasota_yorke()["synthetic_lasota_yorke"] == 1.0
