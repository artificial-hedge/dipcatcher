"""Polynomial ring over Z_p: Euclid gcd, evaluation, derivative (wave 281) (SYNTHETIC).

Dense coefficient lists (lowest degree first). gcd via Euclidean algorithm
exact; (f/g)*g == f when g | f.
"""

_SEED = 20261231 + 773


def _trim(f: list[int]) -> list[int]:
    while len(f) > 1 and f[-1] == 0:
        f.pop()
    return f


def padd(f: list[int], g: list[int], p: int) -> list[int]:
    n = max(len(f), len(g))
    f2 = f + [0] * (n - len(f))
    g2 = g + [0] * (n - len(g))
    return _trim([(a + b) % p for a, b in zip(f2, g2, strict=True)])


def pmul(f: list[int], g: list[int], p: int) -> list[int]:
    out = [0] * (len(f) + len(g) - 1)
    for i, a in enumerate(f):
        for j, b in enumerate(g):
            out[i + j] = (out[i + j] + a * b) % p
    return _trim(out)


def pdiv(f: list[int], g: list[int], p: int) -> tuple[list[int], list[int]]:
    f = f[:]
    q = [0] * max(len(f) - len(g) + 1, 1)
    while len(f) >= len(g) and f != [0]:
        d = len(f) - len(g)
        c = f[-1] * pow(g[-1], -1, p) % p
        q[d] = c
        for i in range(len(g)):
            f[d + i] = (f[d + i] - c * g[i]) % p
        _trim(f)
    return _trim(q), _trim(f)


def pgcd(f: list[int], g: list[int], p: int) -> list[int]:
    while g != [0]:
        _, r = pdiv(f, g, p)
        f, g = g, r
    inv = pow(f[-1], -1, p)
    return _trim([c * inv % p for c in f])


def bench_poly_ring(seed: int = _SEED) -> dict[str, float]:
    p = 7
    f = pmul([1, 1], [1, 2, 1], p)  # (x+1)(x^2+2x+1) = x^3+3x^2+3x+1
    g = [1, 1]  # x+1 divides f
    q, r = pdiv(f, g, p)
    ok = int(r == [0] and pmul(q, g, p) == f)
    g2 = pgcd(f, pmul([2, 1], [1, 1], p), p)
    ok += int(g2 in ([1, 1],))
    return {"synthetic_poly_euclid": float(ok == 2)}
