"""SYNTHETIC point-validation tests for evaluator public keys."""

from __future__ import annotations

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from fx1.ed25519_public import is_canonical_prime_order_public_key

_P = 2**255 - 19

# The eight canonical encodings of Ed25519's small-order subgroup. The first
# is the identity; the others have orders 2, 4, or 8.
_TORSION_ENCODINGS = (
    "0100000000000000000000000000000000000000000000000000000000000000",
    "26e8958fc2b227b045c3f489f2ef98f0d5dfac05d3c63339b13802886d53fc05",
    "0000000000000000000000000000000000000000000000000000000000000000",
    "c7176a703d4dd84fba3c0b760d10670f2a2053fa2c39ccc64ec7fd7792ac037a",
    "ecffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f",
    "c7176a703d4dd84fba3c0b760d10670f2a2053fa2c39ccc64ec7fd7792ac03fa",
    "0000000000000000000000000000000000000000000000000000000000000080",
    "26e8958fc2b227b045c3f489f2ef98f0d5dfac05d3c63339b13802886d53fc85",
)


@pytest.mark.parametrize("seed_byte", (0x11, 0x21, 0x51, 0xA3))
def test_synthetic_generated_public_keys_are_accepted(seed_byte: int) -> None:
    private = Ed25519PrivateKey.from_private_bytes(bytes([seed_byte]) * 32)
    assert is_canonical_prime_order_public_key(private.public_key().public_bytes_raw())


@pytest.mark.parametrize("encoded_hex", _TORSION_ENCODINGS)
def test_synthetic_every_canonical_torsion_point_is_rejected(encoded_hex: str) -> None:
    assert not is_canonical_prime_order_public_key(bytes.fromhex(encoded_hex))


def test_synthetic_noncanonical_and_malformed_encodings_are_rejected() -> None:
    assert not is_canonical_prime_order_public_key(b"")
    assert not is_canonical_prime_order_public_key(b"\x01" * 31)
    assert not is_canonical_prime_order_public_key(b"\x01" * 33)
    assert not is_canonical_prime_order_public_key((_P + 1).to_bytes(32, "little"))
    assert not is_canonical_prime_order_public_key(_P.to_bytes(32, "little"))
    # The identity's x coordinate is zero, so setting its sign bit is invalid.
    assert not is_canonical_prime_order_public_key(bytes.fromhex("01" + "00" * 30 + "80"))


def test_synthetic_normal_point_plus_order_two_torsion_is_rejected() -> None:
    # Adding (0, -1), the order-two point, maps (x, y) to (-x, -y).
    # The result is a valid curve point with a torsion component and must not
    # pass a check that merely rejects the eight pure torsion encodings.
    private = Ed25519PrivateKey.from_private_bytes(b"\x39" * 32)
    ordinary = private.public_key().public_bytes_raw()
    value = int.from_bytes(ordinary, "little")
    y = value & ((1 << 255) - 1)
    x_sign = value >> 255
    assert y not in (0, 1)
    mixed = ((_P - y) % _P | ((x_sign ^ 1) << 255)).to_bytes(32, "little")
    assert mixed != ordinary
    assert is_canonical_prime_order_public_key(ordinary)
    assert not is_canonical_prime_order_public_key(mixed)
