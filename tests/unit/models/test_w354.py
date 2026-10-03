from quant_fund.models.dedekind_check import bench_dedekind_check
from quant_fund.models.divisor_group import bench_divisor_group
from quant_fund.models.genus_riemann import bench_genus_riemann
from quant_fund.models.local_ring_zn import bench_local_ring_zn
from quant_fund.models.moduli_naive import bench_moduli_naive
from quant_fund.models.sheaf_gluing import bench_sheaf_gluing


def test_sheaf_gluing():
    assert bench_sheaf_gluing()["synthetic_sheaf_gluing"] == 1.0


def test_local_ring_zn():
    assert bench_local_ring_zn()["synthetic_local_ring_zn"] == 1.0


def test_dedekind_check():
    assert bench_dedekind_check()["synthetic_dedekind_check"] == 1.0


def test_divisor_group():
    assert bench_divisor_group()["synthetic_divisor_group"] == 1.0


def test_genus_riemann():
    assert bench_genus_riemann()["synthetic_genus_riemann"] == 1.0


def test_moduli_naive():
    assert bench_moduli_naive()["synthetic_moduli_naive"] == 1.0
