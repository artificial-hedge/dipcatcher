"""Wave-241 databases-2 canon tests."""

from __future__ import annotations

from quant_fund.models.aries_recovery import bench_aries_recovery
from quant_fund.models.blink_tree import bench_blink_tree
from quant_fund.models.buffer_pool import bench_buffer_pool
from quant_fund.models.mvcc_gc import bench_mvcc_gc
from quant_fund.models.selinger_join import bench_selinger_join
from quant_fund.models.two_phase_lock import bench_two_phase_lock

_FORBIDDEN = ("sharpe", "sortino", "calmar", "pnl", "nav")


def _clean(out: dict[str, float]) -> None:
    assert not any(b in k.lower() for k in out for b in _FORBIDDEN)
    assert all(v == v and abs(v) < 1e18 for v in out.values())


class TestARIES:
    def test_bench(self) -> None:
        out = bench_aries_recovery()
        _clean(out)
        assert out["synthetic_committed_survives"] == 1.0
        assert out["synthetic_losers_rolled_back"] == 1.0


class Test2PL:
    def test_bench(self) -> None:
        out = bench_two_phase_lock()
        _clean(out)
        assert out["synthetic_conflict_serializable"] == 1.0
        assert out["synthetic_exclusive_writes"] == 1.0


class TestSelinger:
    def test_bench(self) -> None:
        out = bench_selinger_join()
        _clean(out)
        assert out["synthetic_dp_is_optimal"] == 1.0
        assert out["synthetic_order_invariant"] == 1.0


class TestMVCC:
    def test_bench(self) -> None:
        out = bench_mvcc_gc()
        _clean(out)
        assert out["synthetic_snapshots_readable"] == 1.0
        assert out["synthetic_live_version_kept"] == 1.0


class TestBuffer:
    def test_bench(self) -> None:
        out = bench_buffer_pool()
        _clean(out)
        assert out["synthetic_flush_preserves_writes"] == 1.0
        assert out["synthetic_clock_miss_ratio"] < 1.0


class TestBlink:
    def test_bench(self) -> None:
        out = bench_blink_tree()
        _clean(out)
        assert out["synthetic_finds_all"] == 1.0
        assert out["synthetic_sorted_chain"] == 1.0
