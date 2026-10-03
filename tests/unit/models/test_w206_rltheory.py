"""Wave-206 RL-theory canon tests."""

from __future__ import annotations

from quant_fund.models.egreedy_decay import bench_egreedy_decay
from quant_fund.models.mw_hedge import bench_mw_hedge
from quant_fund.models.pi_contraction import bench_pi_contraction
from quant_fund.models.qlearn_rate import bench_qlearn_rate
from quant_fund.models.td_rate import bench_td_rate
from quant_fund.models.ucb_bound import bench_ucb_bound

_FORBIDDEN = ("sharpe", "sortino", "calmar", "pnl", "nav")


def _clean(out: dict[str, float]) -> None:
    assert not any(b in k.lower() for k in out for b in _FORBIDDEN)
    assert all(v == v and abs(v) < 1e18 for v in out.values())


class TestUCBBound:
    def test_bench(self) -> None:
        out = bench_ucb_bound()
        _clean(out)
        assert out["synthetic_ucb_margin"] > 0.0


class TestMWHedge:
    def test_bench(self) -> None:
        out = bench_mw_hedge()
        _clean(out)
        assert out["synthetic_mw_margin"] > 0.0
        assert out["synthetic_mw_rate"] < 5.0


class TestEgreedyDecay:
    def test_bench(self) -> None:
        out = bench_egreedy_decay()
        _clean(out)
        assert out["synthetic_eg_decay_gain"] > 0.0


class TestPIContraction:
    def test_bench(self) -> None:
        out = bench_pi_contraction()
        _clean(out)
        assert out["synthetic_pi_max_ratio"] <= out["synthetic_pi_gamma"] + 1e-9
        assert out["synthetic_pi_v_err"] < 1e-9


class TestTDRate:
    def test_bench(self) -> None:
        out = bench_td_rate()
        _clean(out)
        assert out["synthetic_td_err_late"] < out["synthetic_td_err_early"]


class TestQLearnRate:
    def test_bench(self) -> None:
        out = bench_qlearn_rate()
        _clean(out)
        assert out["synthetic_ql_err_late"] < out["synthetic_ql_err_early"]
