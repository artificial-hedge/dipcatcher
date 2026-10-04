from quant_fund.models.calabi_conjecture import bench_calabi_conjecture
from quant_fund.models.calabi_yau_mfd import bench_calabi_yau_mfd
from quant_fund.models.csck_metric import bench_csck_metric
from quant_fund.models.futaki_invariant import bench_futaki_invariant
from quant_fund.models.k_stability import bench_k_stability
from quant_fund.models.kahler_einstein import bench_kahler_einstein


def test_calabi_yau_mfd():
    assert bench_calabi_yau_mfd()["synthetic_calabi_yau_mfd"] == 1.0


def test_calabi_conjecture():
    assert bench_calabi_conjecture()["synthetic_calabi_conjecture"] == 1.0


def test_kahler_einstein():
    assert bench_kahler_einstein()["synthetic_kahler_einstein"] == 1.0


def test_k_stability():
    assert bench_k_stability()["synthetic_k_stability"] == 1.0


def test_csck_metric():
    assert bench_csck_metric()["synthetic_csck_metric"] == 1.0


def test_futaki_invariant():
    assert bench_futaki_invariant()["synthetic_futaki_invariant"] == 1.0
