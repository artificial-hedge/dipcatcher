"""Wave-232 blockchain canon tests."""

from __future__ import annotations

from quant_fund.models.block_validator import bench_block_validator
from quant_fund.models.difficulty_retarget import bench_difficulty_retarget
from quant_fund.models.fork_resolution import bench_fork_resolution
from quant_fund.models.merkle_tree import bench_merkle_tree
from quant_fund.models.proof_of_work import bench_proof_of_work
from quant_fund.models.utxo_set import bench_utxo_set

_FORBIDDEN = ("sharpe", "sortino", "calmar", "pnl", "nav")


def _clean(out: dict[str, float]) -> None:
    assert not any(b in k.lower() for k in out for b in _FORBIDDEN)
    assert all(v == v and abs(v) < 1e18 for v in out.values())


class TestMerkle:
    def test_bench(self) -> None:
        out = bench_merkle_tree()
        _clean(out)
        assert out["synthetic_proof_valid"] == 1.0
        assert out["synthetic_tamper_detected"] == 1.0


class TestPoW:
    def test_bench(self) -> None:
        out = bench_proof_of_work()
        _clean(out)
        assert out["synthetic_valid_solutions"] == 1.0
        assert out["synthetic_work_near_geometric"] == 1.0


class TestUTXO:
    def test_bench(self) -> None:
        out = bench_utxo_set()
        _clean(out)
        assert out["synthetic_valid_spend"] == 1.0
        assert out["synthetic_double_spend_rejected"] == 1.0


class TestRetarget:
    def test_bench(self) -> None:
        out = bench_difficulty_retarget()
        _clean(out)
        assert out["synthetic_faster_harder"] == 1.0
        assert out["synthetic_clamp_fast"] == 1.0


class TestFork:
    def test_bench(self) -> None:
        out = bench_fork_resolution()
        _clean(out)
        assert out["synthetic_heaviest_wins"] == 1.0
        assert out["synthetic_common_prefix"] == 1.0


class TestBlockVal:
    def test_bench(self) -> None:
        out = bench_block_validator()
        _clean(out)
        assert out["synthetic_valid_accepted"] == 1.0
        assert out["synthetic_bad_prev_rejected"] == 1.0
        assert out["synthetic_bad_pow_rejected"] == 1.0
