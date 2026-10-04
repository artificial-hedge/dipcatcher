from quant_fund.models.band_gerbe import bench_band_gerbe
from quant_fund.models.dm_stack2 import bench_dm_stack2
from quant_fund.models.gerbe2 import bench_gerbe2
from quant_fund.models.inertia_stack import bench_inertia_stack
from quant_fund.models.rigid_stack import bench_rigid_stack
from quant_fund.models.root_stack import bench_root_stack


def test_gerbe2():
    assert bench_gerbe2()["synthetic_gerbe2"] == 1.0


def test_band_gerbe():
    assert bench_band_gerbe()["synthetic_band_gerbe"] == 1.0


def test_rigid_stack():
    assert bench_rigid_stack()["synthetic_rigid_stack"] == 1.0


def test_dm_stack2():
    assert bench_dm_stack2()["synthetic_dm_stack2"] == 1.0


def test_inertia_stack():
    assert bench_inertia_stack()["synthetic_inertia_stack"] == 1.0


def test_root_stack():
    assert bench_root_stack()["synthetic_root_stack"] == 1.0
