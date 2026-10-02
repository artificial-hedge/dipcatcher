"""Wave-214 online-algorithms canon tests."""

from __future__ import annotations

from quant_fund.models.marking_paging import bench_marking_paging
from quant_fund.models.online_gradient import bench_online_gradient
from quant_fund.models.ranking_matching import bench_ranking_matching
from quant_fund.models.secretary_prophet import bench_secretary_prophet
from quant_fund.models.ski_rental import bench_ski_rental
from quant_fund.models.work_function_kserver import bench_work_function_kserver

_FORBIDDEN = ("sharpe", "sortino", "calmar", "pnl", "nav")


def _clean(out: dict[str, float]) -> None:
    assert not any(b in k.lower() for k in out for b in _FORBIDDEN)
    assert all(v == v and abs(v) < 1e18 for v in out.values())


class TestSkiRental:
    def test_bench(self) -> None:
        out = bench_ski_rental()
        _clean(out)
        assert out["synthetic_ski_det_ratio"] <= out["synthetic_ski_det_bound"] + 1e-9
        assert out["synthetic_ski_rand_ratio"] < out["synthetic_ski_det_ratio"]


class TestMarking:
    def test_bench(self) -> None:
        out = bench_marking_paging()
        _clean(out)
        assert out["synthetic_page_marking"] >= out["synthetic_page_belady"]
        assert out["synthetic_page_marking_ratio"] <= out["synthetic_page_k_bound"]


class TestKServer:
    def test_bench(self) -> None:
        out = bench_work_function_kserver()
        _clean(out)
        assert out["synthetic_wfa_online"] >= out["synthetic_wfa_offline"] - 1e-9


class TestRanking:
    def test_bench(self) -> None:
        out = bench_ranking_matching()
        _clean(out)
        assert out["synthetic_rk_ratio"] >= out["synthetic_rk_bound"] - 0.05


class TestSecretary:
    def test_bench(self) -> None:
        out = bench_secretary_prophet()
        _clean(out)
        assert out["synthetic_prophet_ratio"] >= out["synthetic_prophet_bound"]
        assert out["synthetic_sec_hit"] > 0.2


class TestOGD:
    def test_bench(self) -> None:
        out = bench_online_gradient()
        _clean(out)
        assert out["synthetic_ogd_regret_per_sqrt"] < 10.0
