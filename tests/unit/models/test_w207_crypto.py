"""Wave-207 crypto-primitives canon tests."""

from __future__ import annotations

from quant_fund.models.aes_sbox import bench_aes_sbox
from quant_fund.models.diffie_hellman import bench_diffie_hellman
from quant_fund.models.ecc_secp256k1 import bench_ecc_secp256k1
from quant_fund.models.pedersen_commit import bench_pedersen_commit
from quant_fund.models.sha256_impl import bench_sha256_impl
from quant_fund.models.shamir_secret import bench_shamir_secret

_FORBIDDEN = ("sharpe", "sortino", "calmar", "pnl", "nav")


def _clean(out: dict[str, float]) -> None:
    assert not any(b in k.lower() for k in out for b in _FORBIDDEN)
    assert all(v == v and abs(v) < 1e18 for v in out.values())


class TestSHA256:
    def test_bench(self) -> None:
        out = bench_sha256_impl()
        _clean(out)
        assert out["synthetic_sha_kat_match"] == 1.0
        assert out["synthetic_sha_avalanche_bits"] > 100.0


class TestAESSbox:
    def test_bench(self) -> None:
        out = bench_aes_sbox()
        _clean(out)
        assert out["synthetic_sbox_kat"] == 1.0
        assert out["synthetic_sbox_permutation"] == 1.0
        assert out["synthetic_sbox_diff_uniformity"] <= 4.0


class TestShamir:
    def test_bench(self) -> None:
        out = bench_shamir_secret()
        _clean(out)
        assert out["synthetic_shamir_exact"] == out["synthetic_shamir_combos"]
        assert out["synthetic_shamir_under_err"] == 1.0


class TestPedersen:
    def test_bench(self) -> None:
        out = bench_pedersen_commit()
        _clean(out)
        assert out["synthetic_ped_homomorphic"] == 1.0
        assert out["synthetic_ped_open_ok"] == 1.0
        assert out["synthetic_ped_open_bad"] == 0.0


class TestDH:
    def test_bench(self) -> None:
        out = bench_diffie_hellman()
        _clean(out)
        assert out["synthetic_dh_agree"] == 1.0
        assert out["synthetic_dh_ddh"] == 1.0


class TestECC:
    def test_bench(self) -> None:
        out = bench_ecc_secp256k1()
        _clean(out)
        assert out["synthetic_ecc_oncurve_g"] == 1.0
        assert out["synthetic_ecc_2g_x_match"] == 1.0
        assert out["synthetic_ecc_order"] == 1.0
