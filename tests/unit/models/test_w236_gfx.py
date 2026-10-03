"""Wave-236 graphics canon tests."""

from __future__ import annotations

from quant_fund.models.bresenham_line import bench_bresenham_line
from quant_fund.models.bsp_tree import bench_bsp_tree
from quant_fund.models.mvp_transform import bench_mvp_transform
from quant_fund.models.quaternion_slerp import bench_quaternion_slerp
from quant_fund.models.raycaster import bench_raycaster
from quant_fund.models.scanline_fill import bench_scanline_fill
from quant_fund.models.zbuffer_render import bench_zbuffer_render

_FORBIDDEN = ("sharpe", "sortino", "calmar", "pnl", "nav")


def _clean(out: dict[str, float]) -> None:
    assert not any(b in k.lower() for k in out for b in _FORBIDDEN)
    assert all(v == v and abs(v) < 1e18 for v in out.values())


class TestRay:
    def test_bench(self) -> None:
        out = bench_raycaster()
        _clean(out)
        assert out["synthetic_dist_exact"] == 1.0


class TestBres:
    def test_bench(self) -> None:
        out = bench_bresenham_line()
        _clean(out)
        assert out["synthetic_within_half_px"] == 1.0
        assert out["synthetic_8_connected"] == 1.0


class TestFill:
    def test_bench(self) -> None:
        out = bench_scanline_fill()
        _clean(out)
        assert out["synthetic_area_match"] == 1.0


class TestZbuf:
    def test_bench(self) -> None:
        out = bench_zbuffer_render()
        _clean(out)
        assert out["synthetic_nearer_wins"] == 1.0


class TestSlerp:
    def test_bench(self) -> None:
        out = bench_quaternion_slerp()
        _clean(out)
        assert out["synthetic_endpoints"] == 1.0
        assert out["synthetic_bisects_angle"] == 1.0


class TestBSP:
    def test_bench(self) -> None:
        out = bench_bsp_tree()
        _clean(out)
        assert out["synthetic_region_correct"] == 1.0


class TestMVP:
    def test_bench(self) -> None:
        out = bench_mvp_transform()
        _clean(out)
        assert out["synthetic_trs_composed"] == 1.0
        assert out["synthetic_inverse_exact"] == 1.0
