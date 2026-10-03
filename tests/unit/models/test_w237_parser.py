"""Wave-237 parser canon tests."""

from __future__ import annotations

from quant_fund.models.earley_parser import bench_earley_parser
from quant_fund.models.ll1_table import bench_ll1_table
from quant_fund.models.peg_packrat import bench_peg_packrat
from quant_fund.models.pratt_parser import bench_pratt_parser
from quant_fund.models.recursive_descent import bench_recursive_descent
from quant_fund.models.slr_parser import bench_slr_parser

_FORBIDDEN = ("sharpe", "sortino", "calmar", "pnl", "nav")


def _clean(out: dict[str, float]) -> None:
    assert not any(b in k.lower() for k in out for b in _FORBIDDEN)
    assert all(v == v and abs(v) < 1e18 for v in out.values())


class TestRD:
    def test_bench(self) -> None:
        out = bench_recursive_descent()
        _clean(out)
        assert out["synthetic_matches_oracle"] == 1.0
        assert out["synthetic_rejects_bad"] == 1.0


class TestPratt:
    def test_bench(self) -> None:
        out = bench_pratt_parser()
        _clean(out)
        assert out["synthetic_right_assoc_power"] == 1.0


class TestEarley:
    def test_bench(self) -> None:
        out = bench_earley_parser()
        _clean(out)
        assert out["synthetic_parens_correct"] == 1.0


class TestSLR:
    def test_bench(self) -> None:
        out = bench_slr_parser()
        _clean(out)
        assert out["synthetic_shift_reduce_value"] == 1.0


class TestPEG:
    def test_bench(self) -> None:
        out = bench_peg_packrat()
        _clean(out)
        assert out["synthetic_ordered_choice"] == 1.0
        assert out["synthetic_packrat_memo"] == 1.0


class TestLL1:
    def test_bench(self) -> None:
        out = bench_ll1_table()
        _clean(out)
        assert out["synthetic_parses_valid"] == 1.0
        assert out["synthetic_table_complete"] == 1.0
