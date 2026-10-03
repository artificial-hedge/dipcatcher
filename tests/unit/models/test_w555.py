from quant_fund.models.hms_conjecture import bench_hms_conjecture
from quant_fund.models.landau_ginzburg import bench_landau_ginzburg
from quant_fund.models.mirror_functor import bench_mirror_functor
from quant_fund.models.syz_mirror import bench_syz_mirror
from quant_fund.models.torus_fibration import bench_torus_fibration
from quant_fund.models.wrapped_fukaya import bench_wrapped_fukaya


def test_hms_conjecture():
    assert bench_hms_conjecture()["synthetic_hms_conjecture"] == 1.0


def test_landau_ginzburg():
    assert bench_landau_ginzburg()["synthetic_landau_ginzburg"] == 1.0


def test_syz_mirror():
    assert bench_syz_mirror()["synthetic_syz_mirror"] == 1.0


def test_torus_fibration():
    assert bench_torus_fibration()["synthetic_torus_fibration"] == 1.0


def test_wrapped_fukaya():
    assert bench_wrapped_fukaya()["synthetic_wrapped_fukaya"] == 1.0


def test_mirror_functor():
    assert bench_mirror_functor()["synthetic_mirror_functor"] == 1.0
