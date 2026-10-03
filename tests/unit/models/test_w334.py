from quant_fund.models.beaver_triple import bench_beaver_triple
from quant_fund.models.bgw_mpc import bench_bgw_mpc
from quant_fund.models.garbled_circuit import bench_garbled_circuit
from quant_fund.models.ot_extension import bench_ot_extension
from quant_fund.models.psi_intersect import bench_psi_intersect
from quant_fund.models.spdz_mac import bench_spdz_mac


def test_garbled_circuit():
    assert bench_garbled_circuit()["synthetic_garbled_circuit"] == 1.0


def test_bgw_mpc():
    assert bench_bgw_mpc()["synthetic_bgw_mpc"] == 1.0


def test_beaver_triple():
    assert bench_beaver_triple()["synthetic_beaver_triple"] == 1.0


def test_ot_extension():
    assert bench_ot_extension()["synthetic_ot_extension"] == 1.0


def test_spdz_mac():
    assert bench_spdz_mac()["synthetic_spdz_mac"] == 1.0


def test_psi_intersect():
    assert bench_psi_intersect()["synthetic_psi_intersect"] == 1.0
