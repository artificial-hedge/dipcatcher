from quant_fund.models.besicovitch import bench_besicovitch
from quant_fund.models.density_thm import bench_density_thm
from quant_fund.models.marstrand import bench_marstrand
from quant_fund.models.preiss_rect import bench_preiss_rect
from quant_fund.models.rectifiability import bench_rectifiability
from quant_fund.models.tangent_measure import bench_tangent_measure


def test_rectifiability():
    assert bench_rectifiability()["synthetic_rectifiability"] == 1.0


def test_tangent_measure():
    assert bench_tangent_measure()["synthetic_tangent_measure"] == 1.0


def test_density_thm():
    assert bench_density_thm()["synthetic_density_thm"] == 1.0


def test_marstrand():
    assert bench_marstrand()["synthetic_marstrand"] == 1.0


def test_besicovitch():
    assert bench_besicovitch()["synthetic_besicovitch"] == 1.0


def test_preiss_rect():
    assert bench_preiss_rect()["synthetic_preiss_rect"] == 1.0
