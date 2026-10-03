"""Wave-212 approximation-algorithm canon tests."""

from __future__ import annotations

from quant_fund.models.christofides_tsp import bench_christofides_tsp
from quant_fund.models.fptas_knapsack import bench_fptas_knapsack
from quant_fund.models.greedy_set_cover import bench_greedy_set_cover
from quant_fund.models.local_search_maxcut import bench_local_search_maxcut
from quant_fund.models.lp_rounding_sc import bench_lp_rounding_sc
from quant_fund.models.primal_dual_vc import bench_primal_dual_vc

_FORBIDDEN = ("sharpe", "sortino", "calmar", "pnl", "nav")


def _clean(out: dict[str, float]) -> None:
    assert not any(b in k.lower() for k in out for b in _FORBIDDEN)
    assert all(v == v and abs(v) < 1e18 for v in out.values())


class TestGreedySC:
    def test_bench(self) -> None:
        out = bench_greedy_set_cover()
        _clean(out)
        assert out["synthetic_gsc_ratio"] <= out["synthetic_gsc_lnm_bound"]


class TestPrimalDual:
    def test_bench(self) -> None:
        out = bench_primal_dual_vc()
        _clean(out)
        assert out["synthetic_pdvc_ratio"] <= 2.0 + 1e-9
        assert out["synthetic_pdvc_dual_lb"] <= out["synthetic_pdvc_truth"] + 1e-9


class TestLPRounding:
    def test_bench(self) -> None:
        out = bench_lp_rounding_sc()
        _clean(out)
        assert out["synthetic_lpr_feasible"] == 1.0
        assert out["synthetic_lpr_ratio"] <= out["synthetic_lpr_f"] + 1e-9


class TestFPTAS:
    def test_bench(self) -> None:
        out = bench_fptas_knapsack()
        _clean(out)
        assert out["synthetic_fptas_ratio"] >= 0.9


class TestLocalMaxCut:
    def test_bench(self) -> None:
        out = bench_local_search_maxcut()
        _clean(out)
        assert out["synthetic_lsmc_ratio"] >= 0.5
        assert out["synthetic_lsmc_cut"] <= out["synthetic_lsmc_truth"]


class TestChristofides:
    def test_bench(self) -> None:
        out = bench_christofides_tsp()
        _clean(out)
        assert out["synthetic_chr_ratio"] <= 1.5 + 1e-9
        assert out["synthetic_chr_odd"] % 2 == 0
