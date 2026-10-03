"""Wave-240 applied-crypto canon tests."""

from __future__ import annotations

from quant_fund.models.blind_sig import bench_blind_sig
from quant_fund.models.commit_reveal import bench_commit_reveal
from quant_fund.models.merkle_ots import bench_merkle_ots
from quant_fund.models.rsa_toy import bench_rsa_toy
from quant_fund.models.winternitz_ots import bench_winternitz_ots
from quant_fund.models.zkp_schnorr import bench_zkp_schnorr

_FORBIDDEN = ("sharpe", "sortino", "calmar", "pnl", "nav")


def _clean(out: dict[str, float]) -> None:
    assert not any(b in k.lower() for k in out for b in _FORBIDDEN)
    assert all(v == v and abs(v) < 1e18 for v in out.values())


class TestRSA:
    def test_bench(self) -> None:
        out = bench_rsa_toy()
        _clean(out)
        assert out["synthetic_roundtrip"] == 1.0
        assert out["synthetic_tamper_detected"] == 1.0


class TestWOTS:
    def test_bench(self) -> None:
        out = bench_winternitz_ots()
        _clean(out)
        assert out["synthetic_verifies"] == 1.0
        assert out["synthetic_forgery_rejected"] == 1.0


class TestMerkle:
    def test_bench(self) -> None:
        out = bench_merkle_ots()
        _clean(out)
        assert out["synthetic_sig_and_path_valid"] == 1.0
        assert out["synthetic_corrupt_path_fails"] == 1.0


class TestBlind:
    def test_bench(self) -> None:
        out = bench_blind_sig()
        _clean(out)
        assert out["synthetic_unblinds_valid"] == 1.0
        assert out["synthetic_wrong_unblind_fails"] == 1.0


class TestSchnorr:
    def test_bench(self) -> None:
        out = bench_zkp_schnorr()
        _clean(out)
        assert out["synthetic_completeness"] == 1.0
        assert out["synthetic_special_soundness"] == 1.0
        assert out["synthetic_simulator_verifies"] == 1.0


class TestCommit:
    def test_bench(self) -> None:
        out = bench_commit_reveal()
        _clean(out)
        assert out["synthetic_binding"] == 1.0
        assert out["synthetic_open_verifies"] == 1.0
