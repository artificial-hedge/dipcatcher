"""Wave-204 queueing-network + reliability canon tests."""

from __future__ import annotations

from quant_fund.models.bcmp_mva import bench_bcmp_mva
from quant_fund.models.ctmc_availability import bench_ctmc_availability
from quant_fund.models.gordon_newell import bench_gordon_newell
from quant_fund.models.jackson_network import bench_jackson_network
from quant_fund.models.renewal_reward import bench_renewal_reward
from quant_fund.models.vacation_queue import bench_vacation_queue

_FORBIDDEN = ("sharpe", "sortino", "calmar", "pnl", "nav")


def _clean(out: dict[str, float]) -> None:
    assert not any(b in k.lower() for k in out for b in _FORBIDDEN)
    assert all(v == v and abs(v) < 1e18 for v in out.values())


class TestJackson:
    def test_bench(self) -> None:
        out = bench_jackson_network()
        _clean(out)
        assert out["synthetic_jn_stable"] == 1.0
        assert out["synthetic_jn_node0_err"] < 0.15


class TestBCMPMVA:
    def test_bench(self) -> None:
        out = bench_bcmp_mva()
        _clean(out)
        assert out["synthetic_mva_err"] < 1e-9
        assert abs(out["synthetic_mva_total"] - 8.0) < 1e-9


class TestGordonNewell:
    def test_bench(self) -> None:
        out = bench_gordon_newell()
        _clean(out)
        assert out["synthetic_gn_x_err"] < 1e-9
        assert out["synthetic_gn_x_n"] < out["synthetic_gn_bound"]


class TestCTMCAvailability:
    def test_bench(self) -> None:
        out = bench_ctmc_availability()
        _clean(out)
        assert 0.0 < out["synthetic_avail_stat"] < 1.0
        assert out["synthetic_avail_trans_gap"] < 1e-6


class TestRenewalReward:
    def test_bench(self) -> None:
        out = bench_renewal_reward()
        _clean(out)
        assert out["synthetic_rr_err_late"] < out["synthetic_rr_err_early"]


class TestVacationQueue:
    def test_bench(self) -> None:
        out = bench_vacation_queue()
        _clean(out)
        assert out["synthetic_vq_n_err"] < 0.5
        assert out["synthetic_vq_uplift_vs_pk"] > 0.0
