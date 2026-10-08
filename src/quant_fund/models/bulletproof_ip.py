"""Bulletproof-style inner product argument (toy additive-group version) (SYNTHETIC).

Proves <a,b> = v given commitment P = <a,G> + <b,H> + v*U over vectors
of length 2^k. Recursive halving: each round emits (L,R) group elems and
folds a,b by challenge x. Verify recomputes the final check — honest
implementation of the halving recursion on small vectors.
"""

from __future__ import annotations

import random as _r

_SEED = 20261231 + 1050

N = 104729


def _vec_commit(a: list[int], g: list[int], n: int = N) -> int:
    return sum(x * y for x, y in zip(a, g, strict=True)) % n


def _ip(a: list[int], b: list[int], n: int = N) -> int:
    return sum(x * y for x, y in zip(a, b, strict=True)) % n


def _fold(v: list[int], x: int, n: int = N) -> list[int]:
    """Fold vector by challenge: v' = v_lo*x + v_hi*x^-1 style pairing."""
    h = len(v) // 2
    xinv = pow(x, n - 2, n)
    return [(v[i] * x + v[i + h] * xinv) % n for i in range(h)]


def _fold_g(v: list[int], x: int, n: int = N) -> list[int]:
    h = len(v) // 2
    xinv = pow(x, n - 2, n)
    return [(v[i] * xinv + v[i + h] * x) % n for i in range(h)]


def prove(
    a: list[int],
    b: list[int],
    g: list[int],
    h: list[int],
    u: int,
    n: int = N,
    rng: _r.Random | None = None,
) -> dict:
    """Produce IPP transcript: list of (L,R) rounds + final a,b."""
    rng = rng or _r.Random(_SEED)
    rounds = []
    a, b, g, h = a[:], b[:], g[:], h[:]
    while len(a) > 1:
        half = len(a) // 2
        alo, ahi = a[:half], a[half:]
        blo, bhi = b[:half], b[half:]
        glo, ghi = g[:half], g[half:]
        hlo, hhi = h[:half], h[half:]
        cl = _vec_commit(alo, ghi) + _vec_commit(bhi, hlo) + _ip(alo, bhi) * u
        cr = _vec_commit(ahi, glo) + _vec_commit(blo, hhi) + _ip(ahi, blo) * u
        cl %= n
        cr %= n
        rounds.append((cl, cr))
        x = rng.randrange(1, n)
        a = _fold(a, x, n)
        b = _fold_g(b, x, n)
        g = _fold_g(g, x, n)
        h = _fold(h, x, n)
    return {"rounds": rounds, "a": a[0], "b": b[0]}


def verify(
    p_comm: int,
    v: int,
    proof: dict,
    g0: list[int],
    h0: list[int],
    u: int,
    n: int = N,
    rng: _r.Random | None = None,
) -> bool:
    """Re-derive challenges deterministically? For the toy we accept the
    challenge transcript via the same rng stream — verifies the final
    identity P' == a*G' + b*H' + a*b*U after folding."""
    rng = rng or _r.Random(_SEED)
    rounds = proof["rounds"]
    a_fin, b_fin = proof["a"], proof["b"]
    # fold generator vectors through the same challenge stream
    g, h = g0[:], h0[:]
    p = p_comm
    for cl, cr in rounds:
        x = rng.randrange(1, n)
        p = (p + cl * x * x + cr * pow(x, n - 2, n) ** 2) % n
        g = _fold_g(g, x, n)
        h = _fold(h, x, n)
    return bool(p == (a_fin * g[0] + b_fin * h[0] + (a_fin * b_fin % n) * u) % n)


def bench_bulletproof_ip(seed: int = _SEED) -> dict[str, float]:
    rng = _r.Random(seed)
    checks: list[bool] = []
    n = 4
    g = [rng.randrange(1, N) for _ in range(n)]
    h = [rng.randrange(1, N) for _ in range(n)]
    u = rng.randrange(1, N)
    a = [3, 5, 7, 9]
    b = [2, 4, 6, 8]
    v = _ip(a, b)
    p_comm = (_vec_commit(a, g) + _vec_commit(b, h) + v * u) % N
    # use the same rng for prove and verify so challenges match
    prove_rng = _r.Random(777)
    verify_rng = _r.Random(777)
    proof = prove(a, b, g, h, u, rng=prove_rng)
    checks.append(verify(p_comm, v, proof, g, h, u, rng=verify_rng))
    return {"synthetic_bulletproof_ip": float(sum(checks)) / len(checks)}
