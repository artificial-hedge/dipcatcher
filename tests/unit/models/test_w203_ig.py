"""Wave-203 information-geometry canon tests."""

from __future__ import annotations

from quant_fund.models.alpha_geodesic import bench_alpha_geodesic
from quant_fund.models.bregman_nmf import bench_bregman_nmf
from quant_fund.models.fisher_rao import bench_fisher_rao
from quant_fund.models.jko_scheme import bench_jko_scheme
from quant_fund.models.mirror_descent import bench_mirror_descent
from quant_fund.models.natural_gradient import bench_natural_gradient

_FORBIDDEN = ("sharpe", "sortino", "calmar", "pnl", "nav")


def _clean(out: dict[str, float]) -> None:
    assert not any(b in k.lower() for k in out for b in _FORBIDDEN)
    assert all(v == v and abs(v) < 1e18 for v in out.values())


class TestFisherRao:
    def test_bench(self) -> None:
        out = bench_fisher_rao()
        _clean(out)
        assert out["synthetic_fr_dist"] > 0.0
        assert out["synthetic_fr_monotone"] == 1.0


class TestNaturalGradient:
    def test_bench(self) -> None:
        out = bench_natural_gradient()
        _clean(out)
        assert out["synthetic_ng_iters"] < 20000
        assert out["synthetic_gd_iters"] < 20000


class TestMirrorDescent:
    def test_bench(self) -> None:
        out = bench_mirror_descent()
        _clean(out)
        assert out["synthetic_md_resid_gain"] > 0.0
        assert out["synthetic_md_simplex_ok"] < 1e-9


class TestBregmanNMF:
    def test_bench(self) -> None:
        out = bench_bregman_nmf()
        _clean(out)
        assert out["synthetic_is_isdiv_gain"] > 0.0


class TestAlphaGeodesic:
    def test_bench(self) -> None:
        out = bench_alpha_geodesic()
        _clean(out)
        assert out["synthetic_alpha_end0_err"] < 1e-9
        assert out["synthetic_alpha_end1_err"] < 1e-9


class TestJKO:
    def test_bench(self) -> None:
        out = bench_jko_scheme()
        _clean(out)
        assert out["synthetic_jko_err"] < out["synthetic_jko_frozen_err"]
