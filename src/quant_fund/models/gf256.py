"""GF(256) canon: Galois-field arithmetic for Reed–Solomon —
exp/log tables over the RS primitive polynomial 0x11D,
add/mul/div/inverse/power plus polynomial multiply and evaluate.
Bench: field axioms on random triples, inverse correctness, and
generator-polynomial structure used by RS. All SYNTHETIC.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

_PRIMITIVE = 0x11D

_EXP = [0] * 512
_LOG = [0] * 256
_x = 1
for _i in range(255):
    _EXP[_i] = _x
    _LOG[_x] = _i
    _x <<= 1
    if _x & 0x100:
        _x ^= _PRIMITIVE
for _i in range(255, 512):
    _EXP[_i] = _EXP[_i - 255]


def gf_add(a: int, b: int) -> int:
    return a ^ b


def gf_mul(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return _EXP[_LOG[a] + _LOG[b]]


def gf_div(a: int, b: int) -> int:
    if b == 0:
        raise ZeroDivisionError("gf div by 0")
    if a == 0:
        return 0
    return _EXP[(_LOG[a] - _LOG[b]) % 255]


def gf_inv(a: int) -> int:
    if a == 0:
        raise ZeroDivisionError("gf inv of 0")
    return _EXP[255 - _LOG[a]]


def gf_pow(a: int, n: int) -> int:
    if n == 0:
        return 1
    if a == 0:
        return 0
    return _EXP[(_LOG[a] * n) % 255]


def gf_poly_mul(p: list[int], q: list[int]) -> list[int]:
    out = [0] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        for j, b in enumerate(q):
            out[i + j] ^= gf_mul(a, b)
    return out


def gf_poly_eval(p: list[int], x: int) -> int:
    out = 0
    for c in p:
        out = gf_mul(out, x) ^ c
    return out


def bench_gf256(seed: int = 20261231) -> dict[str, float]:
    out: dict[str, float] = {}
    rng = np.random.default_rng(seed)
    n = 400
    viol = 0
    for _ in range(n):
        a = int(rng.integers(1, 256))
        b = int(rng.integers(1, 256))
        c = int(rng.integers(1, 256))
        # multiplicative inverse
        viol += gf_mul(a, gf_inv(a)) != 1
        # distributive law
        viol += gf_mul(a, b ^ c) != (gf_mul(a, b) ^ gf_mul(a, c))
        # associativity
        viol += gf_mul(gf_mul(a, b), c) != gf_mul(a, gf_mul(b, c))
        # division inverse
        viol += gf_div(gf_mul(a, b), b) != a
    out["synthetic_gf_axiom_violations"] = float(viol)
    # log/exp roundtrip on all nonzero
    bad = sum(1 for a in range(1, 256) if _EXP[_LOG[a]] != a)
    out["synthetic_gf_logexp_bad"] = float(bad)
    # power: a^255 = 1 (Fermat)
    bad2 = sum(1 for a in range(1, 256) if gf_pow(a, 255) != 1)
    out["synthetic_gf_fermat_bad"] = float(bad2)
    # generator polynomial for RS(255,239): prod_{i=0}^{15}(x - α^i)
    g = [1]
    for i in range(16):
        g = gf_poly_mul(g, [1, _EXP[i]])
    # g divides x^255 - 1? check g has exactly 17 coefficients
    out["synthetic_gf_genpoly_len"] = float(len(g))
    # evaluate generator at roots → all zero
    root_err = sum(abs(gf_poly_eval(g, _EXP[i])) for i in range(16))
    out["synthetic_gf_genpoly_roots"] = float(root_err)
    return out
