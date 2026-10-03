"""Wave-235 OS-internals canon tests."""

from __future__ import annotations

from quant_fund.models.cfs_scheduler import bench_cfs_scheduler
from quant_fund.models.deadlock_detect import bench_deadlock_detect
from quant_fund.models.demand_paging import bench_demand_paging
from quant_fund.models.disk_sched import bench_disk_sched
from quant_fund.models.fs_journal import bench_fs_journal
from quant_fund.models.round_robin_sched import bench_round_robin_sched

_FORBIDDEN = ("sharpe", "sortino", "calmar", "pnl", "nav")


def _clean(out: dict[str, float]) -> None:
    assert not any(b in k.lower() for k in out for b in _FORBIDDEN)
    assert all(v == v and abs(v) < 1e18 for v in out.values())


class TestRR:
    def test_bench(self) -> None:
        out = bench_round_robin_sched()
        _clean(out)
        assert out["synthetic_sjf_min_tat"] == 1.0
        assert out["synthetic_rr_min_response"] == 1.0


class TestCFS:
    def test_bench(self) -> None:
        out = bench_cfs_scheduler()
        _clean(out)
        assert out["synthetic_fair_share"] == 1.0
        assert out["synthetic_proportional_weight"] == 1.0


class TestPaging:
    def test_bench(self) -> None:
        out = bench_demand_paging()
        _clean(out)
        assert out["synthetic_opt_min_faults"] == 1.0
        assert out["synthetic_lru_near_opt"] == 1.0


class TestDeadlock:
    def test_bench(self) -> None:
        out = bench_deadlock_detect()
        _clean(out)
        assert out["synthetic_cycle_detected"] == 1.0
        assert out["synthetic_bankers_oracle"] == 1.0


class TestDisk:
    def test_bench(self) -> None:
        out = bench_disk_sched()
        _clean(out)
        assert out["synthetic_sstf_le_fcfs"] == 1.0
        assert out["synthetic_scan_monotone"] == 1.0


class TestJournal:
    def test_bench(self) -> None:
        out = bench_fs_journal()
        _clean(out)
        assert out["synthetic_committed_visible"] == 1.0
        assert out["synthetic_idempotent_replay"] == 1.0
