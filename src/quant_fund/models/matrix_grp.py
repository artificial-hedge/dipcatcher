"""Matrix groups over GF(p): order formulas and subgroup checks (wave 281) (SYNTHETIC).

|GL(n,p)| = prod_{i=0}^{n-1} (p^n - p^i); |SL| = |GL| / (p-1). Verified by
brute-force enumeration of invertible 2x2 matrices mod p for small p, plus
subgroup-closure check for SL.
"""

import numpy as np

_SEED = 20261231 + 775


def _det2(m: np.ndarray, p: int) -> int:
    return int(m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]) % p


def gl_order_enum(p: int) -> tuple[int, int]:
    gl, sl = 0, 0
    for a in range(p):
        for b in range(p):
            for c in range(p):
                for d in range(p):
                    det = (a * d - b * c) % p
                    if det != 0:
                        gl += 1
                        if det == 1:
                            sl += 1
    return gl, sl


def _sl_closed(p: int) -> bool:
    mats = [
        np.array([[a, b], [c, d]])
        for a in range(p)
        for b in range(p)
        for c in range(p)
        for d in range(p)
        if (a * d - b * c) % p == 1
    ]
    for m1 in mats:
        for m2 in mats:
            if _det2((m1 @ m2) % p, p) != 1:
                return False
    return True


def bench_matrix_grp(seed: int = _SEED) -> dict[str, float]:
    p = 3
    gl, sl = gl_order_enum(p)
    want_gl = (p**2 - 1) * (p**2 - p)
    ok = int(gl == want_gl)
    ok += int(sl == want_gl // (p - 1))
    ok += int(_sl_closed(p))
    return {"synthetic_matrix_grp": float(ok == 3)}
