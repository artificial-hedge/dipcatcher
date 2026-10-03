"""Wave-231 architecture canon tests."""

from __future__ import annotations

from quant_fund.models.branch_predictor import bench_branch_predictor
from quant_fund.models.cache_sim import bench_cache_sim
from quant_fund.models.cpu_pipeline import bench_cpu_pipeline
from quant_fund.models.paging_sim import bench_paging_sim
from quant_fund.models.roofline_model import bench_roofline_model
from quant_fund.models.tomasulo_sim import bench_tomasulo_sim

_FORBIDDEN = ("sharpe", "sortino", "calmar", "pnl", "nav")


def _clean(out: dict[str, float]) -> None:
    assert not any(b in k.lower() for k in out for b in _FORBIDDEN)
    assert all(v == v and abs(v) < 1e18 for v in out.values())


class TestPipeline:
    def test_bench(self) -> None:
        out = bench_cpu_pipeline()
        _clean(out)
        assert out["synthetic_regs_match_seq"] == 1.0
        assert out["synthetic_fwd_never_worse"] == 1.0
        assert out["synthetic_cpi_ge_1"] == 1.0


class TestCache:
    def test_bench(self) -> None:
        out = bench_cache_sim()
        _clean(out)
        assert out["synthetic_cold_misses"] == 1.0
        assert out["synthetic_lru_order"] == 1.0
        assert out["synthetic_capacity_monotone"] == 1.0


class TestBranch:
    def test_bench(self) -> None:
        out = bench_branch_predictor()
        _clean(out)
        assert out["synthetic_beats_always_taken"] == 1.0


class TestTomasulo:
    def test_bench(self) -> None:
        out = bench_tomasulo_sim()
        _clean(out)
        assert out["synthetic_regs_match_inorder"] == 1.0


class TestPaging:
    def test_bench(self) -> None:
        out = bench_paging_sim()
        _clean(out)
        assert out["synthetic_tlb_locality"] == 1.0
        assert out["synthetic_fault_count_ok"] == 1.0


class TestRoofline:
    def test_bench(self) -> None:
        out = bench_roofline_model()
        _clean(out)
        assert out["synthetic_ridge_exact"] == 1.0
        assert out["synthetic_never_exceeds"] == 1.0
