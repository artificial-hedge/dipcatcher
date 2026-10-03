"""Wave-224 string-algorithm canon tests."""

from __future__ import annotations

from quant_fund.models.aho_corasick import bench_aho_corasick
from quant_fund.models.bwt_transform import bench_bwt_transform
from quant_fund.models.edit_distance import bench_edit_distance
from quant_fund.models.kmp_search import bench_kmp_search
from quant_fund.models.lz77 import bench_lz77
from quant_fund.models.suffix_automaton import bench_suffix_automaton

_FORBIDDEN = ("sharpe", "sortino", "calmar", "pnl", "nav")


def _clean(out: dict[str, float]) -> None:
    assert not any(b in k.lower() for k in out for b in _FORBIDDEN)
    assert all(v == v and abs(v) < 1e18 for v in out.values())


class TestAho:
    def test_bench(self) -> None:
        out = bench_aho_corasick()
        _clean(out)
        assert out["synthetic_agree"] == 1.0


class TestSAM:
    def test_bench(self) -> None:
        out = bench_suffix_automaton()
        _clean(out)
        assert out["synthetic_distinct_agree"] == 1.0
        assert out["synthetic_count_agree"] == 1.0
        assert out["synthetic_lcs_agree"] == 1.0


class TestKMP:
    def test_bench(self) -> None:
        out = bench_kmp_search()
        _clean(out)
        assert out["synthetic_agree"] == 1.0
        assert out["synthetic_overlap"] == 1.0


class TestEdit:
    def test_bench(self) -> None:
        out = bench_edit_distance()
        _clean(out)
        assert out["synthetic_agree"] == 1.0
        assert out["synthetic_script_valid"] == 1.0


class TestLZ77:
    def test_bench(self) -> None:
        out = bench_lz77()
        _clean(out)
        assert out["synthetic_roundtrip"] == 1.0
        assert out["synthetic_repetitive_ratio"] < 0.2


class TestBWT:
    def test_bench(self) -> None:
        out = bench_bwt_transform()
        _clean(out)
        assert out["synthetic_roundtrip"] == 1.0
