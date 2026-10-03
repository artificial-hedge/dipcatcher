from quant_fund.models.decidable_theory import bench_decidable_theory
from quant_fund.models.definable_set import bench_definable_set
from quant_fund.models.indiscernible_seq import bench_indiscernible_seq
from quant_fund.models.interpol_thm import bench_interpol_thm
from quant_fund.models.omitting_prime import bench_omitting_prime
from quant_fund.models.saturated_model import bench_saturated_model


def test_decidable_theory():
    assert bench_decidable_theory()["synthetic_decidable_theory"] == 1.0


def test_indiscernible_seq():
    assert bench_indiscernible_seq()["synthetic_indiscernible_seq"] == 1.0


def test_saturated_model():
    assert bench_saturated_model()["synthetic_saturated_model"] == 1.0


def test_omitting_prime():
    assert bench_omitting_prime()["synthetic_omitting_prime"] == 1.0


def test_interpol_thm():
    assert bench_interpol_thm()["synthetic_interpol_thm"] == 1.0


def test_definable_set():
    assert bench_definable_set()["synthetic_definable_set"] == 1.0
