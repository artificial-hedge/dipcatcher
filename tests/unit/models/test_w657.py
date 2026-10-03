from quant_fund.models.bousfield_period import bench_bousfield_period
from quant_fund.models.completion_htpy import bench_completion_htpy
from quant_fund.models.homotopy_cartesian import bench_homotopy_cartesian
from quant_fund.models.p_local_htpy import bench_p_local_htpy
from quant_fund.models.ravenel_htpy import bench_ravenel_htpy
from quant_fund.models.snake_constr import bench_snake_constr


def test_ravenel_htpy():
    assert bench_ravenel_htpy()["synthetic_ravenel_htpy"] == 1.0


def test_bousfield_period():
    assert bench_bousfield_period()["synthetic_bousfield_period"] == 1.0


def test_snake_constr():
    assert bench_snake_constr()["synthetic_snake_constr"] == 1.0


def test_homotopy_cartesian():
    assert bench_homotopy_cartesian()["synthetic_homotopy_cartesian"] == 1.0


def test_p_local_htpy():
    assert bench_p_local_htpy()["synthetic_p_local_htpy"] == 1.0


def test_completion_htpy():
    assert bench_completion_htpy()["synthetic_completion_htpy"] == 1.0
