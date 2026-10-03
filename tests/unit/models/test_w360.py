from quant_fund.models.energy_method import bench_energy_method
from quant_fund.models.fundamental_laplace import bench_fundamental_laplace
from quant_fund.models.heat_kernel import bench_heat_kernel
from quant_fund.models.maximum_principle import bench_maximum_principle
from quant_fund.models.wave_dalembert import bench_wave_dalembert
from quant_fund.models.weak_solution import bench_weak_solution


def test_energy_method():
    assert bench_energy_method()["synthetic_energy_method"] == 1.0


def test_maximum_principle():
    assert bench_maximum_principle()["synthetic_maximum_principle"] == 1.0


def test_heat_kernel():
    assert bench_heat_kernel()["synthetic_heat_kernel"] == 1.0


def test_wave_dalembert():
    assert bench_wave_dalembert()["synthetic_wave_dalembert"] == 1.0


def test_weak_solution():
    assert bench_weak_solution()["synthetic_weak_solution"] == 1.0


def test_fundamental_laplace():
    assert bench_fundamental_laplace()["synthetic_fundamental_laplace"] == 1.0
