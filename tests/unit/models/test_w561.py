from quant_fund.models.almgren_pitts import bench_almgren_pitts
from quant_fund.models.brakke_flow import bench_brakke_flow
from quant_fund.models.minimal_surface import bench_minimal_surface
from quant_fund.models.plateau_problem import bench_plateau_problem
from quant_fund.models.simon_regularity import bench_simon_regularity
from quant_fund.models.stable_minimal import bench_stable_minimal


def test_minimal_surface():
    assert bench_minimal_surface()["synthetic_minimal_surface"] == 1.0


def test_plateau_problem():
    assert bench_plateau_problem()["synthetic_plateau_problem"] == 1.0


def test_brakke_flow():
    assert bench_brakke_flow()["synthetic_brakke_flow"] == 1.0


def test_almgren_pitts():
    assert bench_almgren_pitts()["synthetic_almgren_pitts"] == 1.0


def test_simon_regularity():
    assert bench_simon_regularity()["synthetic_simon_regularity"] == 1.0


def test_stable_minimal():
    assert bench_stable_minimal()["synthetic_stable_minimal"] == 1.0
