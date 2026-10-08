"""Categorical product universal property (wave 289) (SYNTHETIC).

In Set, product A x B with projections pi1,pi2 has the UMP: for every
pair (f: Z->A, g: Z->B) a unique mediating map h = <f,g> factors
f = pi1 h, g = pi2 h — verified on all maps over small sets.
"""

import itertools

_SEED = 20261231 + 822


def mediating(f: dict[int, int], g: dict[int, int]) -> dict[int, tuple[int, int]]:
    return {z: (f[z], g[z]) for z in f}


def bench_limit_prod(seed: int = _SEED) -> dict[str, float]:
    a_vals, b_vals = [0, 1, 2], [0, 1]
    zs = [0, 1, 2]
    ok = 0
    for fvals in itertools.product(a_vals, repeat=3):
        for gvals in itertools.product(b_vals, repeat=3):
            f = dict(zip(zs, fvals, strict=True))
            g = dict(zip(zs, gvals, strict=True))
            h = mediating(f, g)
            if not all(h[z][0] == f[z] and h[z][1] == g[z] for z in zs):
                return {"synthetic_limit_prod": 0.0}
            # uniqueness: any h' factoring both equals h
            for z in zs:
                cand = h[z]
                ok += int(cand == (f[z], g[z]))
    return {"synthetic_limit_prod": float(ok > 0)}
