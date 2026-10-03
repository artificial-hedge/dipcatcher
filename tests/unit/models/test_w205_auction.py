"""Wave-205 auction-theory canon tests."""

from __future__ import annotations

from quant_fund.models.all_pay_auction import bench_all_pay_auction
from quant_fund.models.ascending_clock import bench_ascending_clock
from quant_fund.models.double_auction import bench_double_auction
from quant_fund.models.first_price_auction import bench_first_price_auction
from quant_fund.models.gsp_auction import bench_gsp_auction
from quant_fund.models.vickrey_auction import bench_vickrey_auction

_FORBIDDEN = ("sharpe", "sortino", "calmar", "pnl", "nav")


def _clean(out: dict[str, float]) -> None:
    assert not any(b in k.lower() for k in out for b in _FORBIDDEN)
    assert all(v == v and abs(v) < 1e18 for v in out.values())


class TestVickrey:
    def test_bench(self) -> None:
        out = bench_vickrey_auction()
        _clean(out)
        assert out["synthetic_vickrey_err"] < 0.05
        assert out["synthetic_vickrey_equiv_gap"] < 0.05


class TestFirstPrice:
    def test_bench(self) -> None:
        out = bench_first_price_auction()
        _clean(out)
        assert out["synthetic_fpa_err"] < 0.02
        assert out["synthetic_fpa_naive_premium"] > 0.0


class TestAllPay:
    def test_bench(self) -> None:
        out = bench_all_pay_auction()
        _clean(out)
        assert out["synthetic_apa_err"] < 0.05


class TestAscendingClock:
    def test_bench(self) -> None:
        out = bench_ascending_clock()
        _clean(out)
        assert out["synthetic_clock_vickrey_gap"] == 0.0
        assert out["synthetic_clock_theory_gap"] < 0.05


class TestDoubleAuction:
    def test_bench(self) -> None:
        out = bench_double_auction()
        _clean(out)
        assert out["synthetic_da_volume"] > 0.0
        assert out["synthetic_da_gap_volume"] == 0.0


class TestGSP:
    def test_bench(self) -> None:
        out = bench_gsp_auction()
        _clean(out)
        assert out["synthetic_gsp_premium"] >= 0.0
        assert out["synthetic_gsp_envy_viol"] <= 0.0
