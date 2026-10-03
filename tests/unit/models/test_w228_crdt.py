"""Wave-228 CRDT canon tests."""

from __future__ import annotations

from quant_fund.models.gcounter import bench_gcounter
from quant_fund.models.lww_map import bench_lww_map
from quant_fund.models.orset import bench_orset
from quant_fund.models.pncounter import bench_pncounter
from quant_fund.models.rga_sequence import bench_rga_sequence
from quant_fund.models.twopset import bench_twopset

_FORBIDDEN = ("sharpe", "sortino", "calmar", "pnl", "nav")


def _clean(out: dict[str, float]) -> None:
    assert not any(b in k.lower() for k in out for b in _FORBIDDEN)
    assert all(v == v and abs(v) < 1e18 for v in out.values())


class TestGCounter:
    def test_bench(self) -> None:
        out = bench_gcounter()
        _clean(out)
        assert out["synthetic_commutative"] == 1.0
        assert out["synthetic_associative"] == 1.0
        assert out["synthetic_converges"] == 1.0


class TestPNCounter:
    def test_bench(self) -> None:
        out = bench_pncounter()
        _clean(out)
        assert out["synthetic_commutative"] == 1.0
        assert out["synthetic_net_value"] == 1.0


class TestORSet:
    def test_bench(self) -> None:
        out = bench_orset()
        _clean(out)
        assert out["synthetic_commutative"] == 1.0
        assert out["synthetic_add_wins"] == 1.0
        assert out["synthetic_unobserved_noop"] == 1.0


class TestLWW:
    def test_bench(self) -> None:
        out = bench_lww_map()
        _clean(out)
        assert out["synthetic_commutative"] == 1.0
        assert out["synthetic_lww_wins"] == 1.0


class TestTwoPSet:
    def test_bench(self) -> None:
        out = bench_twopset()
        _clean(out)
        assert out["synthetic_remove_wins"] == 1.0
        assert out["synthetic_no_readd"] == 1.0


class TestRGA:
    def test_bench(self) -> None:
        out = bench_rga_sequence()
        _clean(out)
        assert out["synthetic_convergent"] == 1.0
        assert out["synthetic_all_orders"] == 1.0
