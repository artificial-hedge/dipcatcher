from quant_fund.models.homotopy_factor import bench_homotopy_factor
from quant_fund.models.homotopy_fixed import bench_homotopy_fixed
from quant_fund.models.homotopy_lift import bench_homotopy_lift
from quant_fund.models.homotopy_orbit import bench_homotopy_orbit
from quant_fund.models.stable_operad import bench_stable_operad
from quant_fund.models.stable_sheaf import bench_stable_sheaf


def test_homotopy_lift():
    assert bench_homotopy_lift()["synthetic_homotopy_lift"] == 1.0


def test_homotopy_orbit():
    assert bench_homotopy_orbit()["synthetic_homotopy_orbit"] == 1.0


def test_homotopy_fixed():
    assert bench_homotopy_fixed()["synthetic_homotopy_fixed"] == 1.0


def test_stable_operad():
    assert bench_stable_operad()["synthetic_stable_operad"] == 1.0


def test_homotopy_factor():
    assert bench_homotopy_factor()["synthetic_homotopy_factor"] == 1.0


def test_stable_sheaf():
    assert bench_stable_sheaf()["synthetic_stable_sheaf"] == 1.0
