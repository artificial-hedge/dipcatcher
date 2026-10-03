"""Wave-218 advanced-derivatives canon tests."""

from __future__ import annotations

from quant_fund.models.andreasen_huge import bench_andreasen_huge
from quant_fund.models.barrier_adjoint import bench_barrier_adjoint
from quant_fund.models.deep_hedge import bench_deep_hedge
from quant_fund.models.dupire_localvol import bench_dupire_localvol
from quant_fund.models.heston_calib import bench_heston_calib
from quant_fund.models.sabr_calib import bench_sabr_calib

_FORBIDDEN = ("sharpe", "sortino", "calmar", "pnl", "nav")


def _clean(out: dict[str, float]) -> None:
    assert not any(b in k.lower() for k in out for b in _FORBIDDEN)
    assert all(v == v and abs(v) < 1e18 for v in out.values())


class TestDupire:
    def test_bench(self) -> None:
        out = bench_dupire_localvol()
        _clean(out)
        assert out["synthetic_dupire_relerr"] < 0.05


class TestSABR:
    def test_bench(self) -> None:
        out = bench_sabr_calib()
        _clean(out)
        assert out["synthetic_sabr_alpha_err"] < 0.05
        assert out["synthetic_sabr_rho_err"] < 0.15


class TestDeepHedge:
    def test_bench(self) -> None:
        out = bench_deep_hedge()
        _clean(out)
        # trained policy matches or beats raw BS delta by construction
        assert out["synthetic_dh_cvar"] >= out["synthetic_dh_bs_cvar"] - 1e-6


class TestHeston:
    def test_bench(self) -> None:
        out = bench_heston_calib()
        _clean(out)
        assert out["synthetic_heston_rho_err"] < 0.1
        assert out["synthetic_heston_px_rmse"] < 0.05


class TestBarrier:
    def test_bench(self) -> None:
        out = bench_barrier_adjoint()
        _clean(out)
        assert out["synthetic_adj_gap"] < 0.05
        assert out["synthetic_adj_dbarrier"] < 0.0  # up barrier -> lower price


class TestAH:
    def test_bench(self) -> None:
        out = bench_andreasen_huge()
        _clean(out)
        assert out["synthetic_ah_mono"] == 1.0
        assert out["synthetic_ah_convex"] == 1.0
