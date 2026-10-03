"""Wave-230 computational-geometry canon tests."""

from __future__ import annotations

from quant_fund.models.closest_pair import bench_closest_pair
from quant_fund.models.ear_clipping import bench_ear_clipping
from quant_fund.models.point_in_polygon import bench_point_in_polygon
from quant_fund.models.rotating_calipers import bench_rotating_calipers
from quant_fund.models.segment_intersection import bench_segment_intersection
from quant_fund.models.sutherland_hodgman import bench_sutherland_hodgman

_FORBIDDEN = ("sharpe", "sortino", "calmar", "pnl", "nav")


def _clean(out: dict[str, float]) -> None:
    assert not any(b in k.lower() for k in out for b in _FORBIDDEN)
    assert all(v == v and abs(v) < 1e18 for v in out.values())


class TestEarClip:
    def test_bench(self) -> None:
        out = bench_ear_clipping()
        _clean(out)
        assert out["synthetic_area_conserved"] == 1.0
        assert out["synthetic_triangle_count"] == 1.0


class TestSH:
    def test_bench(self) -> None:
        out = bench_sutherland_hodgman()
        _clean(out)
        assert out["synthetic_inside_window"] == 1.0
        assert out["synthetic_area_nonincrease"] == 1.0


class TestIsect:
    def test_bench(self) -> None:
        out = bench_segment_intersection()
        _clean(out)
        assert out["synthetic_count_ok"] == 1.0
        assert out["synthetic_point_on_both"] == 1.0


class TestPIP:
    def test_bench(self) -> None:
        out = bench_point_in_polygon()
        _clean(out)
        assert out["synthetic_ray_vs_winding"] == 1.0
        assert out["synthetic_mc_area_ok"] == 1.0


class TestCP:
    def test_bench(self) -> None:
        out = bench_closest_pair()
        _clean(out)
        assert out["synthetic_matches_brute"] == 1.0


class TestCalipers:
    def test_bench(self) -> None:
        out = bench_rotating_calipers()
        _clean(out)
        assert out["synthetic_diameter_ok"] == 1.0
        assert out["synthetic_hull_convex"] == 1.0
