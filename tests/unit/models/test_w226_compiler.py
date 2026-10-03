"""Wave-226 compiler/formal-language canon tests."""

from __future__ import annotations

from quant_fund.models.cyk_parser import bench_cyk_parser
from quant_fund.models.dfa_minimize import bench_dfa_minimize
from quant_fund.models.dominance_tree import bench_dominance_tree
from quant_fund.models.linscan_regalloc import bench_linscan_regalloc
from quant_fund.models.liveness_dce import bench_liveness_dce
from quant_fund.models.regex_engine import bench_regex_engine

_FORBIDDEN = ("sharpe", "sortino", "calmar", "pnl", "nav")


def _clean(out: dict[str, float]) -> None:
    assert not any(b in k.lower() for k in out for b in _FORBIDDEN)
    assert all(v == v and abs(v) < 1e18 for v in out.values())


class TestRegexEngine:
    def test_bench(self) -> None:
        out = bench_regex_engine()
        _clean(out)
        assert out["synthetic_agree"] == 1.0
        assert out["synthetic_n_tests"] >= 200


class TestDFAMinimize:
    def test_bench(self) -> None:
        out = bench_dfa_minimize()
        _clean(out)
        assert out["synthetic_partition_ok"] == 1.0
        assert out["synthetic_lang_equiv"] == 1.0


class TestCYK:
    def test_bench(self) -> None:
        out = bench_cyk_parser()
        _clean(out)
        assert out["synthetic_agree"] == 1.0
        assert out["synthetic_known_members"] == 1.0


class TestDominance:
    def test_bench(self) -> None:
        out = bench_dominance_tree()
        _clean(out)
        assert out["synthetic_dom_agree"] == 1.0
        assert out["synthetic_idom_tree"] == 1.0
        assert out["synthetic_recurrence"] == 1.0


class TestLiveness:
    def test_bench(self) -> None:
        out = bench_liveness_dce()
        _clean(out)
        assert out["synthetic_live_agree"] == 1.0
        assert out["synthetic_same_output"] == 1.0
        assert out["synthetic_idempotent"] == 1.0


class TestLinscan:
    def test_bench(self) -> None:
        out = bench_linscan_regalloc()
        _clean(out)
        assert out["synthetic_no_overlap"] == 1.0
        assert out["synthetic_spill_consistent"] == 1.0
        assert out["synthetic_forced_spills"] == 1.0
