from quant_fund.models.bockstein_ss import bench_bockstein_ss
from quant_fund.models.bousfield_ss import bench_bousfield_ss
from quant_fund.models.cartan_ss import bench_cartan_ss
from quant_fund.models.eilenberg_moore import (
    bench_eilenberg_moore,
)
from quant_fund.models.lyndon_ss import bench_lyndon_ss
from quant_fund.models.serre_ss4 import bench_serre_ss4


def test_serre_ss4():
    assert bench_serre_ss4()["synthetic_serre_ss4"] == 1.0


def test_bockstein_ss():
    assert bench_bockstein_ss()["synthetic_bockstein_ss"] == 1.0


def test_eilenberg_moore():
    assert bench_eilenberg_moore()["synthetic_eilenberg_moore"] == 1.0


def test_bousfield_ss():
    assert bench_bousfield_ss()["synthetic_bousfield_ss"] == 1.0


def test_lyndon_ss():
    assert bench_lyndon_ss()["synthetic_lyndon_ss"] == 1.0


def test_cartan_ss():
    assert bench_cartan_ss()["synthetic_cartan_ss"] == 1.0
