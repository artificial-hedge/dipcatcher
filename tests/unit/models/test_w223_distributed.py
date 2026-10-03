"""Wave-223 distributed-systems canon tests."""

from __future__ import annotations

from quant_fund.models.consistent_hash import bench_consistent_hash
from quant_fund.models.gossip_epidemic import bench_gossip_epidemic
from quant_fund.models.paxos import bench_paxos
from quant_fund.models.pbft_lite import bench_pbft_lite
from quant_fund.models.raft_election import bench_raft_election
from quant_fund.models.vector_clock import bench_vector_clock

_FORBIDDEN = ("sharpe", "sortino", "calmar", "pnl", "nav")


def _clean(out: dict[str, float]) -> None:
    assert not any(b in k.lower() for k in out for b in _FORBIDDEN)
    assert all(v == v and abs(v) < 1e18 for v in out.values())


class TestPaxos:
    def test_bench(self) -> None:
        out = bench_paxos()
        _clean(out)
        assert out["synthetic_agree"] == 1.0
        assert out["synthetic_valid"] == 1.0
        assert out["synthetic_deterministic"] == 1.0


class TestRaft:
    def test_bench(self) -> None:
        out = bench_raft_election()
        _clean(out)
        assert out["synthetic_safety"] == 1.0
        assert out["synthetic_liveness"] >= 0.9


class TestVectorClock:
    def test_bench(self) -> None:
        out = bench_vector_clock()
        _clean(out)
        assert out["synthetic_agree"] == 1.0
        assert out["synthetic_concurrent"] == 1.0


class TestConsistentHash:
    def test_bench(self) -> None:
        out = bench_consistent_hash()
        _clean(out)
        assert out["synthetic_minimal_remap"] == 1.0
        assert out["synthetic_moved_frac"] < 0.3


class TestGossip:
    def test_bench(self) -> None:
        out = bench_gossip_epidemic()
        _clean(out)
        assert out["synthetic_coverage"] == 1.0
        assert out["synthetic_partial_exact"] == 1.0


class TestPBFT:
    def test_bench(self) -> None:
        out = bench_pbft_lite()
        _clean(out)
        assert out["synthetic_safety"] == 1.0
        assert out["synthetic_deterministic"] == 1.0
