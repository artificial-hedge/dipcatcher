from quant_fund.models.cardy_formula import bench_cardy_formula
from quant_fund.models.duminil_copin import bench_duminil_copin
from quant_fund.models.grimmett_percolation import (
    bench_grimmett_percolation,
)
from quant_fund.models.kesten_percolation import (
    bench_kesten_percolation,
)
from quant_fund.models.russo_seymour import bench_russo_seymour
from quant_fund.models.smirnov_percolation import (
    bench_smirnov_percolation,
)


def test_smirnov_percolation():
    assert bench_smirnov_percolation()["synthetic_smirnov_percolation"] == 1.0


def test_duminil_copin():
    assert bench_duminil_copin()["synthetic_duminil_copin"] == 1.0


def test_kesten_percolation():
    assert bench_kesten_percolation()["synthetic_kesten_percolation"] == 1.0


def test_cardy_formula():
    assert bench_cardy_formula()["synthetic_cardy_formula"] == 1.0


def test_russo_seymour():
    assert bench_russo_seymour()["synthetic_russo_seymour"] == 1.0


def test_grimmett_percolation():
    assert bench_grimmett_percolation()["synthetic_grimmett_percolation"] == 1.0
