from quant_fund.models.chelkak_ising import bench_chelkak_ising
from quant_fund.models.duminil_copin2 import (
    bench_duminil_copin2,
)
from quant_fund.models.hongler_ising import bench_hongler_ising
from quant_fund.models.kenyon_dimers import bench_kenyon_dimers
from quant_fund.models.smirnov_ising import bench_smirnov_ising
from quant_fund.models.thurston_tiling import (
    bench_thurston_tiling,
)


def test_smirnov_ising():
    assert bench_smirnov_ising()["synthetic_smirnov_ising"] == 1.0


def test_chelkak_ising():
    assert bench_chelkak_ising()["synthetic_chelkak_ising"] == 1.0


def test_kenyon_dimers():
    assert bench_kenyon_dimers()["synthetic_kenyon_dimers"] == 1.0


def test_thurston_tiling():
    assert bench_thurston_tiling()["synthetic_thurston_tiling"] == 1.0


def test_duminil_copin2():
    assert bench_duminil_copin2()["synthetic_duminil_copin2"] == 1.0


def test_hongler_ising():
    assert bench_hongler_ising()["synthetic_hongler_ising"] == 1.0
