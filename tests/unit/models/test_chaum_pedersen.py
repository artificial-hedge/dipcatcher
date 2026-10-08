"""Chaum-Pedersen proof determinism tests."""

from __future__ import annotations

import secrets

from quant_fund.models.chaum_pedersen import _p, bench_chaum_pedersen, cp_prove, cp_verify


def test_prove_accepts_injected_nonce_draw():
    p = _p()
    x = 12345
    proof = cp_prove(5, 7, pow(5, x, p), pow(7, x, p), x, randbelow=lambda b: 42)
    assert cp_verify(5, 7, pow(5, x, p), pow(7, x, p), proof)


def test_prove_deterministic_under_fixed_nonce():
    p = _p()
    x = 777
    y, z = pow(5, x, p), pow(7, x, p)
    p1 = cp_prove(5, 7, y, z, x, randbelow=lambda b: 9)
    p2 = cp_prove(5, 7, y, z, x, randbelow=lambda b: 9)
    assert p1 == p2


def test_bench_uses_seeded_rng_not_secrets(monkeypatch):
    """Old code drew x and the nonce from ``secrets.randbelow`` — a seeded
    bench must not touch the OS CSPRNG at all."""

    def _boom(bound: int) -> int:
        raise AssertionError("bench reached for secrets.randbelow")

    monkeypatch.setattr(secrets, "randbelow", _boom)
    out = bench_chaum_pedersen(seed=3)
    assert out["synthetic_cp_valid"] == 1.0


def test_bench_deterministic():
    assert bench_chaum_pedersen(seed=5) == bench_chaum_pedersen(seed=5)


def test_verify_rejects_forged():
    p = _p()
    x = 4242
    y, z = pow(5, x, p), pow(7, x, p)
    proof = cp_prove(5, 7, y, z, x, randbelow=lambda b: 11)
    a1, a2, r = proof
    # forged z (different exponent) must fail verification
    assert not cp_verify(5, 7, y, pow(7, x + 1, p), (a1, a2, r))
