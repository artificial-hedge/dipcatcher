"""Unit tests for quant_fund.models._crypto_synth."""

from __future__ import annotations

import hashlib

from quant_fund.models._crypto_synth import (
    AES_SBOX_KNOWN,
    G_SAFE,
    P_SAFE,
    SECP_GX,
    SECP_GY,
    SECP_N,
    SECP_P,
    SHA_EXPECT,
    SHA_MSG,
)


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True


def test_sha256_kat() -> None:
    assert hashlib.sha256(SHA_MSG).hexdigest() == SHA_EXPECT
    # the recorded answer must be the real digest, not a placeholder
    assert SHA_EXPECT == "a396e72bb0423be13f52e61b25de1492c801ded5846024a1488236d6cf98873e"


def test_safe_prime_and_generator_order() -> None:
    assert P_SAFE == 2 * 419 + 1
    assert _is_prime(P_SAFE) and _is_prime(419)
    # G_SAFE must generate the order-q quadratic-residue subgroup
    assert pow(G_SAFE, 419, P_SAFE) == 1
    assert G_SAFE != 1 and G_SAFE != P_SAFE - 1
    # and it must be a quadratic residue (Legendre = 1)
    assert pow(G_SAFE, (P_SAFE - 1) // 2, P_SAFE) == 1


def test_aes_sbox_known_values() -> None:
    # ground-truth AES S-box entries
    assert AES_SBOX_KNOWN[0x00] == 0x63
    assert AES_SBOX_KNOWN[0x01] == 0x7C
    assert AES_SBOX_KNOWN[0x53] == 0xED
    assert AES_SBOX_KNOWN[0xFF] == 0x16


def test_secp256k1_generator_on_curve() -> None:
    # y^2 == x^3 + 7 (mod p)
    assert (SECP_GY * SECP_GY - (pow(SECP_GX, 3, SECP_P) + 7)) % SECP_P == 0
    # N*G must be the point at infinity: verify order divides N via
    # checking SECP_N is the documented subgroup order
    assert SECP_N == 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141


def test_sha_expect_is_not_trivially_wrong() -> None:
    assert len(SHA_EXPECT) == 64
    int(SHA_EXPECT, 16)
