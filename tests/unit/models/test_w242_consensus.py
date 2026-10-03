"""Wave-242 consensus/distributed-2 canon tests."""

from __future__ import annotations

from quant_fund.models.epaxos import bench_epaxos
from quant_fund.models.multi_paxos import bench_multi_paxos
from quant_fund.models.swim_gossip import bench_swim_gossip
from quant_fund.models.two_three_pc import bench_two_three_pc
from quant_fund.models.viewstamped import bench_viewstamped
from quant_fund.models.zab_protocol import bench_zab_protocol

_FORBIDDEN = ("sharpe", "sortino", "calmar", "pnl", "nav")


def _clean(out: dict[str, float]) -> None:
    assert not any(b in k.lower() for k in out for b in _FORBIDDEN)
    assert all(v == v and abs(v) < 1e18 for v in out.values())


class TestMultiPaxos:
    def test_bench(self) -> None:
        out = bench_multi_paxos()
        _clean(out)
        assert out["synthetic_single_value_per_slot"] == 1.0
        assert out["synthetic_all_slots_decided"] == 1.0


class TestEPaxos:
    def test_bench(self) -> None:
        out = bench_epaxos()
        _clean(out)
        assert out["synthetic_replicas_agree"] == 1.0
        assert out["synthetic_deterministic_order"] == 1.0


class TestVR:
    def test_bench(self) -> None:
        out = bench_viewstamped()
        _clean(out)
        assert out["synthetic_view_change_covers_committed"] == 1.0
        assert out["synthetic_primary_log_valid"] == 1.0


class TestZAB:
    def test_bench(self) -> None:
        out = bench_zab_protocol()
        _clean(out)
        assert out["synthetic_total_order"] == 1.0
        assert out["synthetic_no_gaps"] == 1.0


class TestSWIM:
    def test_bench(self) -> None:
        out = bench_swim_gossip()
        _clean(out)
        assert out["synthetic_converges"] == 1.0
        assert out["synthetic_suspect_propagates"] == 1.0


class Test2PC:
    def test_bench(self) -> None:
        out = bench_two_three_pc()
        _clean(out)
        assert out["synthetic_agreement"] == 1.0
        assert out["synthetic_validity"] == 1.0
