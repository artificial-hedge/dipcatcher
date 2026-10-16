"""Standalone mirror of quant_fund.utils.ed25519_public.

The auditor scripts cannot import the implementation they audit. Keep this
small, pure-Python predicate in lockstep with the library through the parity
test in test_standalone_ed25519_public.py.
"""

from __future__ import annotations

_P = 2**255 - 19
_L = 2**252 + 27742317777372353535851937790883648493
_D = (-121665 * pow(121666, _P - 2, _P)) % _P
_SQRT_M1 = pow(2, (_P - 1) // 4, _P)

type _Point = tuple[int, int, int, int]
_IDENTITY: _Point = (0, 1, 1, 0)


def _decode(encoded: bytes) -> _Point | None:
    if len(encoded) != 32:
        return None
    value = int.from_bytes(encoded, "little")
    sign = value >> 255
    y = value & ((1 << 255) - 1)
    if y >= _P:
        return None

    y_squared = y * y % _P
    denominator = (_D * y_squared + 1) % _P
    if denominator == 0:
        return None
    x_squared = (y_squared - 1) * pow(denominator, _P - 2, _P) % _P
    x = pow(x_squared, (_P + 3) // 8, _P)
    if x * x % _P != x_squared:
        x = x * _SQRT_M1 % _P
    if x * x % _P != x_squared or (x == 0 and sign):
        return None
    if x & 1 != sign:
        x = _P - x
    return (x, y, 1, x * y % _P)


def _add(left: _Point, right: _Point) -> _Point:
    x1, y1, z1, t1 = left
    x2, y2, z2, t2 = right
    a = (y1 - x1) * (y2 - x2) % _P
    b = (y1 + x1) * (y2 + x2) % _P
    c = 2 * _D * t1 * t2 % _P
    d = 2 * z1 * z2 % _P
    e = (b - a) % _P
    f = (d - c) % _P
    g = (d + c) % _P
    h = (b + a) % _P
    return (e * f % _P, g * h % _P, f * g % _P, e * h % _P)


def _multiply_by_group_order(point: _Point) -> _Point:
    result = _IDENTITY
    addend = point
    scalar = _L
    while scalar:
        if scalar & 1:
            result = _add(result, addend)
        addend = _add(addend, addend)
        scalar >>= 1
    return result


def _is_identity(point: _Point) -> bool:
    x, y, z, _ = point
    return x % _P == 0 and (y - z) % _P == 0


def is_canonical_prime_order_public_key(encoded: bytes) -> bool:
    """Require a canonical, nonidentity point in the prime-order subgroup."""
    point = _decode(encoded)
    return (
        point is not None
        and not _is_identity(point)
        and _is_identity(_multiply_by_group_order(point))
    )
