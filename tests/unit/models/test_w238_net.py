"""Wave-238 networking canon tests."""

from __future__ import annotations

from quant_fund.models.http2_flow import bench_http2_flow
from quant_fund.models.nat_table import bench_nat_table
from quant_fund.models.rtt_estimator import bench_rtt_estimator
from quant_fund.models.sliding_window import bench_sliding_window
from quant_fund.models.tcp_aimd import bench_tcp_aimd
from quant_fund.models.token_bucket import bench_token_bucket

_FORBIDDEN = ("sharpe", "sortino", "calmar", "pnl", "nav")


def _clean(out: dict[str, float]) -> None:
    assert not any(b in k.lower() for k in out for b in _FORBIDDEN)
    assert all(v == v and abs(v) < 1e18 for v in out.values())


class TestAIMD:
    def test_bench(self) -> None:
        out = bench_tcp_aimd()
        _clean(out)
        assert out["synthetic_slow_start_doubles"] == 1.0
        assert out["synthetic_loss_halves"] == 1.0


class TestARQ:
    def test_bench(self) -> None:
        out = bench_sliding_window()
        _clean(out)
        assert out["synthetic_all_delivered"] == 1.0
        assert out["synthetic_sr_le_gbn_sends"] == 1.0


class TestBucket:
    def test_bench(self) -> None:
        out = bench_token_bucket()
        _clean(out)
        assert out["synthetic_burst_exact"] == 1.0
        assert out["synthetic_longrun_rate"] == 1.0


class TestRTT:
    def test_bench(self) -> None:
        out = bench_rtt_estimator()
        _clean(out)
        assert out["synthetic_tracks_step"] == 1.0
        assert out["synthetic_first_rto_3r"] == 1.0


class TestNAT:
    def test_bench(self) -> None:
        out = bench_nat_table()
        _clean(out)
        assert out["synthetic_no_collisions"] == 1.0
        assert out["synthetic_reverse_lookup"] == 1.0


class TestH2:
    def test_bench(self) -> None:
        out = bench_http2_flow()
        _clean(out)
        assert out["synthetic_window_bounded"] == 1.0
        assert out["synthetic_conn_cap"] == 1.0
