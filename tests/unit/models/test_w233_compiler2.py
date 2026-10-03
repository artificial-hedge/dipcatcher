"""Wave-233 compiler-2 canon tests."""

from __future__ import annotations

from quant_fund.models.gvn_elim import bench_gvn_elim
from quant_fund.models.instr_sched import bench_instr_sched
from quant_fund.models.licm_hoist import bench_licm_hoist
from quant_fund.models.reg_coalesce import bench_reg_coalesce
from quant_fund.models.sccp_const import bench_sccp_const
from quant_fund.models.ssa_construct import bench_ssa_construct

_FORBIDDEN = ("sharpe", "sortino", "calmar", "pnl", "nav")


def _clean(out: dict[str, float]) -> None:
    assert not any(b in k.lower() for k in out for b in _FORBIDDEN)
    assert all(v == v and abs(v) < 1e18 for v in out.values())


class TestSSA:
    def test_bench(self) -> None:
        out = bench_ssa_construct()
        _clean(out)
        assert out["synthetic_single_def"] == 1.0
        assert out["synthetic_phi_at_merge"] == 1.0


class TestSCCP:
    def test_bench(self) -> None:
        out = bench_sccp_const()
        _clean(out)
        assert out["synthetic_consts_match_oracle"] == 1.0
        assert out["synthetic_full_coverage"] == 1.0


class TestGVN:
    def test_bench(self) -> None:
        out = bench_gvn_elim()
        _clean(out)
        assert out["synthetic_results_identical"] == 1.0
        assert out["synthetic_never_grows"] == 1.0


class TestCoalesce:
    def test_bench(self) -> None:
        out = bench_reg_coalesce()
        _clean(out)
        assert out["synthetic_semantics_preserved"] == 1.0
        assert out["synthetic_no_moves_left"] == 1.0


class TestSched:
    def test_bench(self) -> None:
        out = bench_instr_sched()
        _clean(out)
        assert out["synthetic_within_bounds"] == 1.0


class TestLICM:
    def test_bench(self) -> None:
        out = bench_licm_hoist()
        _clean(out)
        assert out["synthetic_result_identical"] == 1.0
        assert out["synthetic_invariant_detected"] == 1.0
