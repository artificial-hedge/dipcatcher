from quant_fund.models.bass_heller_swan import (
    bench_bass_heller_swan,
)
from quant_fund.models.k0_group import bench_k0_group
from quant_fund.models.k1_group import bench_k1_group
from quant_fund.models.k_theory_spec import bench_k_theory_spec
from quant_fund.models.milnor_k2 import bench_milnor_k2
from quant_fund.models.quillen_q import bench_quillen_q


def test_k0_group():
    assert bench_k0_group()["synthetic_k0_group"] == 1.0


def test_k1_group():
    assert bench_k1_group()["synthetic_k1_group"] == 1.0


def test_milnor_k2():
    assert bench_milnor_k2()["synthetic_milnor_k2"] == 1.0


def test_quillen_q():
    assert bench_quillen_q()["synthetic_quillen_q"] == 1.0


def test_k_theory_spec():
    assert bench_k_theory_spec()["synthetic_k_theory_spec"] == 1.0


def test_bass_heller_swan():
    assert bench_bass_heller_swan()["synthetic_bass_heller_swan"] == 1.0
