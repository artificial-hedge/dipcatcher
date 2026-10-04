from quant_fund.models.arithmetic_dm import bench_arithmetic_dm
from quant_fund.models.frobenius_dm import bench_frobenius_dm
from quant_fund.models.holonomic_dm import bench_holonomic_dm
from quant_fund.models.isocrystal import bench_isocrystal
from quant_fund.models.overconv_dm import bench_overconv_dm
from quant_fund.models.rigid_dm import bench_rigid_dm


def test_overconv_dm():
    assert bench_overconv_dm()["synthetic_overconv_dm"] == 1.0


def test_arithmetic_dm():
    assert bench_arithmetic_dm()["synthetic_arithmetic_dm"] == 1.0


def test_frobenius_dm():
    assert bench_frobenius_dm()["synthetic_frobenius_dm"] == 1.0


def test_holonomic_dm():
    assert bench_holonomic_dm()["synthetic_holonomic_dm"] == 1.0


def test_rigid_dm():
    assert bench_rigid_dm()["synthetic_rigid_dm"] == 1.0


def test_isocrystal():
    assert bench_isocrystal()["synthetic_isocrystal"] == 1.0
