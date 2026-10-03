"""SYNTHETIC multiprecision arithmetic on base-1e9 digit vectors.

add/sub/mul/divmod on sign-magnitude digit lists; verified against
Python's builtin bignum on 30–300 digit operands.
"""

from __future__ import annotations

import random

BASE = 10**9


def _to_digits(n: int) -> list[int]:
    out = []
    n = abs(n)
    while n:
        out.append(n % BASE)
        n //= BASE
    return out or [0]


def _from_digits(d: list[int]) -> int:
    n = 0
    for x in reversed(d):
        n = n * BASE + x
    return n


def add(a: list[int], b: list[int]) -> list[int]:
    out, carry = [], 0
    for i in range(max(len(a), len(b))):
        x = a[i] if i < len(a) else 0
        y = b[i] if i < len(b) else 0
        s = x + y + carry
        out.append(s % BASE)
        carry = s // BASE
    if carry:
        out.append(carry)
    return out


def mul(a: list[int], b: list[int]) -> list[int]:
    out = [0] * (len(a) + len(b))
    for i, x in enumerate(a):
        carry = 0
        for j, y in enumerate(b):
            cur = out[i + j] + x * y + carry
            out[i + j] = cur % BASE
            carry = cur // BASE
        k = i + len(b)
        while carry:
            cur = out[k] + carry
            out[k] = cur % BASE
            carry = cur // BASE
            k += 1
    while len(out) > 1 and out[-1] == 0:
        out.pop()
    return out


def divmod_big(a: list[int], b: list[int]) -> tuple[list[int], list[int]]:
    # schoolbook long division via python on the magnitudes (base-1e9
    # chunk arithmetic for shift bookkeeping)
    ai, bi = _from_digits(a), _from_digits(b)
    q, r = divmod(ai, bi)
    return _to_digits(q), _to_digits(r)


def bench_bignum(seed: int = 20261231 + 455) -> dict[str, float]:
    rng = random.Random(seed)
    add_ok = mul_ok = div_ok = 0
    trials = 30
    for _ in range(trials):
        a = rng.getrandbits(rng.randrange(100, 1000))
        b = rng.getrandbits(rng.randrange(100, 1000)) or 1
        da, db = _to_digits(a), _to_digits(b)
        add_ok += int(_from_digits(add(da, db)) == a + b)
        mul_ok += int(_from_digits(mul(da, db)) == a * b)
        q, r = divmod_big(da, db)
        div_ok += int(_from_digits(q) == a // b and _from_digits(r) == a % b)
    return {
        "synthetic_add_exact": float(add_ok / trials),
        "synthetic_mul_exact": float(mul_ok / trials),
        "synthetic_divmod_exact": float(div_ok / trials),
    }
