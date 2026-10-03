from quant_fund.models.karoubi_v import bench_karoubi_v
from quant_fund.models.kv_theory import bench_kv_theory
from quant_fund.models.nk_theory import bench_nk_theory
from quant_fund.models.plus_k import bench_plus_k
from quant_fund.models.vorst_stab import bench_vorst_stab
from quant_fund.models.waldhausen_k import bench_waldhausen_k


def test_waldhausen_k():
    assert bench_waldhausen_k()["synthetic_waldhausen_k"] == 1.0


def test_plus_k():
    assert bench_plus_k()["synthetic_plus_k"] == 1.0


def test_kv_theory():
    assert bench_kv_theory()["synthetic_kv_theory"] == 1.0


def test_karoubi_v():
    assert bench_karoubi_v()["synthetic_karoubi_v"] == 1.0


def test_vorst_stab():
    assert bench_vorst_stab()["synthetic_vorst_stab"] == 1.0


def test_nk_theory():
    assert bench_nk_theory()["synthetic_nk_theory"] == 1.0
