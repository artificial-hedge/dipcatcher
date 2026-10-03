from quant_fund.models.auslander_buchs import bench_auslander_buchs
from quant_fund.models.betti_series import bench_betti_series
from quant_fund.models.green_koszul import bench_green_koszul
from quant_fund.models.minimal_free import bench_minimal_free
from quant_fund.models.quillen_suslin import bench_quillen_suslin
from quant_fund.models.serre_conj import bench_serre_conj


def test_betti_series():
    assert bench_betti_series()["synthetic_betti_series"] == 1.0


def test_minimal_free():
    assert bench_minimal_free()["synthetic_minimal_free"] == 1.0


def test_auslander_buchs():
    assert bench_auslander_buchs()["synthetic_auslander_buchs"] == 1.0


def test_serre_conj():
    assert bench_serre_conj()["synthetic_serre_conj"] == 1.0


def test_quillen_suslin():
    assert bench_quillen_suslin()["synthetic_quillen_suslin"] == 1.0


def test_green_koszul():
    assert bench_green_koszul()["synthetic_green_koszul"] == 1.0
