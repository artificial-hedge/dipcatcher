"""Wave-211 integer-programming canon tests."""

from __future__ import annotations

from quant_fund.models.benders_decomp import bench_benders_decomp
from quant_fund.models.branch_and_cut import bench_branch_and_cut
from quant_fund.models.column_generation import bench_column_generation
from quant_fund.models.gomory_cut import bench_gomory_cut
from quant_fund.models.held_karp import bench_held_karp
from quant_fund.models.lagrangian_relax import bench_lagrangian_relax

_FORBIDDEN = ("sharpe", "sortino", "calmar", "pnl", "nav")


def _clean(out: dict[str, float]) -> None:
    assert not any(b in k.lower() for k in out for b in _FORBIDDEN)
    assert all(v == v and abs(v) < 1e18 for v in out.values())


class TestGomory:
    def test_bench(self) -> None:
        out = bench_gomory_cut()
        _clean(out)
        assert out["synthetic_gom_gap"] < 1e-6
        assert out["synthetic_gom_integral"] == 1.0


class TestColGen:
    def test_bench(self) -> None:
        out = bench_column_generation()
        _clean(out)
        assert out["synthetic_cg_lp_bound"] > 400.0
        assert out["synthetic_cg_bound_vs_naive"] > 0.0


class TestBenders:
    def test_bench(self) -> None:
        out = bench_benders_decomp()
        _clean(out)
        assert out["synthetic_ben_gap"] < 1e-6
        assert out["synthetic_ben_cuts"] > 0.0


class TestLagrangian:
    def test_bench(self) -> None:
        out = bench_lagrangian_relax()
        _clean(out)
        assert out["synthetic_lag_lb"] <= out["synthetic_lag_truth"] + 1e-6
        assert out["synthetic_lag_gap_frac"] < 0.2


class TestBnC:
    def test_bench(self) -> None:
        out = bench_branch_and_cut()
        _clean(out)
        assert out["synthetic_bnc_gap"] < 1e-6
        assert out["synthetic_bnc_nodes_saved"] > 0.0


class TestHeldKarp:
    def test_bench(self) -> None:
        out = bench_held_karp()
        _clean(out)
        assert out["synthetic_hk_bound"] <= out["synthetic_hk_truth"] + 1e-6
        assert out["synthetic_hk_gap_frac"] < 0.1
