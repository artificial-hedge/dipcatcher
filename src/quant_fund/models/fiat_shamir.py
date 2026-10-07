"""Fiat-Shamir non-interactive Schnorr: sigma + RO hash transcript (SYNTHETIC)."""

import hashlib

import numpy as np

_SEED = 20261231 + 694

_P = 524287  # prime (2^19-1)
_Q = _P - 1  # work in full group for toy
_G = 5


def _h(*xs: int) -> int:
    m = hashlib.sha256("|".join(map(str, xs)).encode()).digest()
    return int.from_bytes(m[:8], "big") % _Q


def prove(x: int, r: int) -> tuple[int, int, int]:
    y = pow(_G, x, _P)
    a = pow(_G, r, _P)
    e = _h(y, a)
    z = (r + e * x) % _Q
    return y, a, z


def verify(y: int, a: int, z: int) -> bool:
    e = _h(y, a)
    return pow(_G, z, _P) == (a * pow(y, e, _P)) % _P


def bench_fiat_shamir(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0.0
    trials = 30
    for _ in range(trials):
        x = int(rng.randint(1, _Q - 1))
        r = int(rng.randint(1, _Q - 1))
        y, a, z = prove(x, r)
        ok += float(verify(y, a, z))
        # wrong witness must fail
        y_bad = pow(_G, (x + 1) % _Q, _P)
        ok += float(not verify(y_bad, a, z))
    return {"synthetic_fs_sound": ok / (2 * trials)}
