"""OR-composition of Schnorr proofs (CDS 1994): prove knowledge of one of
two discrete logs without revealing which."""

import hashlib
import secrets

_SEED = 20261231 + 595


def _p() -> int:
    return 2**31 - 1  # toy prime


def _H(*xs: int) -> int:
    h = hashlib.sha256(b"".join(x.to_bytes(8, "big") for x in xs)).digest()
    return int.from_bytes(h[:8], "big") % _p()


def or_proof_prove(g: int, h: int, y1: int, y2: int, wit: int, which: int) -> tuple:
    p = _p()
    c = [0, 0]
    r = [0, 0]
    a = [0, 0]
    c[1 - which] = secrets.randbelow(p - 2)
    r[1 - which] = secrets.randbelow(p - 2)
    bases = [g, h]
    a[1 - which] = (
        pow(bases[1 - which], r[1 - which], p)
        * pow([y1, y2][1 - which], -c[1 - which] % (p - 1), p)
    ) % p
    w = secrets.randbelow(p - 2)
    a[which] = pow(bases[which], w, p)
    e = _H(g, h, y1, y2, a[0], a[1])
    c[which] = (e - c[1 - which]) % (p - 1)
    r[which] = (w + c[which] * wit) % (p - 1)
    return a, c, r


def or_proof_verify(g: int, h: int, y1: int, y2: int, proof: tuple) -> bool:
    a, c, r = proof
    p = _p()
    if (c[0] + c[1]) % (p - 1) != _H(g, h, y1, y2, a[0], a[1]):
        return False
    for i, y in enumerate([y1, y2]):
        b = [g, h][i]
        if pow(b, r[i], p) != (a[i] * pow(y, c[i], p)) % p:
            return False
    return True


def bench_sigma_or_proof(seed: int = 0) -> dict[str, float]:
    import random

    random.seed(seed + _SEED)
    p = _p()
    g, h = 5, 7
    ok = 0
    for _ in range(40):
        x1, x2 = secrets.randbelow(p - 2), secrets.randbelow(p - 2)
        y1, y2 = pow(g, x1, p), pow(h, x2, p)
        which = secrets.randbelow(2)
        wit = [x1, x2][which]
        proof = or_proof_prove(g, h, y1, y2, wit, which)
        ok += or_proof_verify(g, h, y1, y2, proof)
    return {"synthetic_or_proof_valid": ok / 40}
