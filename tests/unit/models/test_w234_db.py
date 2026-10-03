"""Wave-234 database-internals canon tests."""

from __future__ import annotations

from quant_fund.models.btree_index import bench_btree_index
from quant_fund.models.join_algos import bench_join_algos
from quant_fund.models.lsm_tree import bench_lsm_tree
from quant_fund.models.mvcc_isolation import bench_mvcc_isolation
from quant_fund.models.query_planner import bench_query_planner
from quant_fund.models.wal_recovery import bench_wal_recovery

_FORBIDDEN = ("sharpe", "sortino", "calmar", "pnl", "nav")


def _clean(out: dict[str, float]) -> None:
    assert not any(b in k.lower() for k in out for b in _FORBIDDEN)
    assert all(v == v and abs(v) < 1e18 for v in out.values())


class TestBTree:
    def test_bench(self) -> None:
        out = bench_btree_index()
        _clean(out)
        assert out["synthetic_find_exact"] == 1.0
        assert out["synthetic_scan_sorted"] == 1.0
        assert out["synthetic_balanced_depth"] == 1.0


class TestWAL:
    def test_bench(self) -> None:
        out = bench_wal_recovery()
        _clean(out)
        assert out["synthetic_committed_visible"] == 1.0
        assert out["synthetic_uncommitted_gone"] == 1.0


class TestJoins:
    def test_bench(self) -> None:
        out = bench_join_algos()
        _clean(out)
        assert out["synthetic_three_way_agree"] == 1.0


class TestPlanner:
    def test_bench(self) -> None:
        out = bench_query_planner()
        _clean(out)
        assert out["synthetic_index_when_selective"] == 1.0
        assert out["synthetic_greedy_is_optimal"] == 1.0


class TestMVCC:
    def test_bench(self) -> None:
        out = bench_mvcc_isolation()
        _clean(out)
        assert out["synthetic_snapshot_stable"] == 1.0


class TestLSM:
    def test_bench(self) -> None:
        out = bench_lsm_tree()
        _clean(out)
        assert out["synthetic_latest_wins"] == 1.0
        assert out["synthetic_runs_sorted"] == 1.0
