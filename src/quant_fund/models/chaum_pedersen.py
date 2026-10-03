"""Chaum-Pedersen DLOG-equality proof: prove log_g(y) == log_h(z)."""

import hashlib
import secrets

_SEED = 20261231 + 594


def _p() -> int:
    return 2**31 - 1


def _H(*xs: int) -> int:
    d = hashlib.sha256(b"".join(x.to_bytes(8, "big") for x in xs)).digest()
    return int.from_bytes(d[:8], "big") % (_p() - 1)


def cp_prove(g: int, h: int, y: int, z: int, x: int) -> tuple[int, int, int]:
    p = _p()
    w = secrets.randbelow(p - 2)
    a1, a2 = pow(g, w, p), pow(h, w, p)
    c = _H(g, h, y, z, a1, a2)
    r = (w + c * x) % (p - 1)
    return a1, a2, r


def cp_verify(g: int, h: int, y: int, z: int, proof: tuple[int, int, int]) -> bool:
    a1, a2, r = proof
    p = _p()
    c = _H(g, h, y, z, a1, a2)
    return pow(g, r, p) == (a1 * pow(y, c, p)) % p and pow(h, r, p) == (a2 * pow(z, c, p)) % p


def bench_chaum_pedersen(seed: int = _SEED) -> dict[str, float]:
    import random

    random.seed(seed)
    p = _p()
    g, h = 5, 7
    ok = 0
    for _ in range(40):
        x = secrets.randbelow(p - 2)
        y, z = pow(g, x, p), pow(h, x, p)
        proof = cp_prove(g, h, y, z, x)
        ok += cp_verify(g, h, y, z, proof)
    return {"synthetic_cp_valid": ok / 40}
