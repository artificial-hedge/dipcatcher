"""Wave-221 computational-algebra canon tests."""

from __future__ import annotations

from quant_fund.models.buchberger import bench_buchberger
from quant_fund.models.gf2_factor import bench_gf2_factor
from quant_fund.models.lll_reduce import bench_lll_reduce
from quant_fund.models.newton_interp import bench_newton_interp
from quant_fund.models.poly_gcd import bench_poly_gcd
from quant_fund.models.resultant import bench_resultant

_FORBIDDEN = ("sharpe", "sortino", "calmar", "pnl", "nav")


def _clean(out: dict[str, float]) -> None:
    assert not any(b in k.lower() for k in out for b in _FORBIDDEN)
    assert all(v == v and abs(v) < 1e18 for v in out.values())


class TestBuchberger:
    def test_bench(self) -> None:
        out = bench_buchberger()
        _clean(out)
        assert out["synthetic_member_reduces"] == 1.0
        assert out["synthetic_nonmember_stays"] == 1.0


class TestResultant:
    def test_bench(self) -> None:
        out = bench_resultant()
        _clean(out)
        assert out["synthetic_agree"] == 1.0
        assert out["synthetic_disc3"] == 4.0


class TestPolyGCD:
    def test_bench(self) -> None:
        out = bench_poly_gcd()
        _clean(out)
        assert out["synthetic_gcd_correct"] == 1.0
        assert out["synthetic_agree"] == 1.0


class TestGF2Factor:
    def test_bench(self) -> None:
        out = bench_gf2_factor()
        _clean(out)
        assert out["synthetic_product_ok"] == 1.0
        assert out["synthetic_agree"] == 1.0


class TestLLL:
    def test_bench(self) -> None:
        out = bench_lll_reduce()
        _clean(out)
        assert out["synthetic_lovasz"] == 1.0
        assert out["synthetic_within_bound"] == 1.0


class TestNewtonInterp:
    def test_bench(self) -> None:
        out = bench_newton_interp()
        _clean(out)
        assert out["synthetic_agree"] == 1.0
        assert out["synthetic_max_resid"] == 0.0
