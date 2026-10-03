from quant_fund.models.ball_tree import (
    bench_ball_tree,
)
from quant_fund.models.cover_tree import (
    bench_cover_tree,
)
from quant_fund.models.kd_tree import (
    bench_kd_tree,
)
from quant_fund.models.quad_tree import (
    bench_quad_tree,
)
from quant_fund.models.r_tree import (
    bench_r_tree,
)
from quant_fund.models.vp_tree import (
    bench_vp_tree,
)


def test_kd_tree():
    assert bench_kd_tree()["synthetic_kd_tree"] == 1.0


def test_ball_tree():
    assert bench_ball_tree()["synthetic_ball_tree"] == 1.0


def test_cover_tree():
    assert bench_cover_tree()["synthetic_cover_tree"] == 1.0


def test_r_tree():
    assert bench_r_tree()["synthetic_r_tree"] == 1.0


def test_quad_tree():
    assert bench_quad_tree()["synthetic_quad_tree"] == 1.0


def test_vp_tree():
    assert bench_vp_tree()["synthetic_vp_tree"] == 1.0
