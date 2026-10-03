from quant_fund.models.beilinson_height import (
    bench_beilinson_height,
)
from quant_fund.models.brown_motives import bench_brown_motives
from quant_fund.models.mixed_elliptic import (
    bench_mixed_elliptic,
)
from quant_fund.models.motivic_pi import bench_motivic_pi
from quant_fund.models.mzc_motive import bench_mzc_motive
from quant_fund.models.zeta_element import bench_zeta_element


def test_brown_motives():
    assert bench_brown_motives()["synthetic_brown_motives"] == 1.0


def test_mzc_motive():
    assert bench_mzc_motive()["synthetic_mzc_motive"] == 1.0


def test_zeta_element():
    assert bench_zeta_element()["synthetic_zeta_element"] == 1.0


def test_mixed_elliptic():
    assert bench_mixed_elliptic()["synthetic_mixed_elliptic"] == 1.0


def test_motivic_pi():
    assert bench_motivic_pi()["synthetic_motivic_pi"] == 1.0


def test_beilinson_height():
    assert bench_beilinson_height()["synthetic_beilinson_height"] == 1.0
