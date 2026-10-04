"""Koszul complex over Z/nZ on a short sequence: homology = 0 for regular seq (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def koszul_differentials(seq: list[int], mod: int) -> tuple[np.ndarray, np.ndarray]:
    """K(x1,x2): 0 -> R -d2-> R^2 -d1-> R -> 0 over Z/mod.
    d2(r) = (x2*r, -x1*r); d1(a,b) = x1*a + x2*b."""
    d2 = np.array([[seq[1] % mod], [(-seq[0]) % mod]], dtype=int)
    d1 = np.array([[seq[0] % mod, seq[1] % mod]], dtype=int)
    return d2, d1


def z_rank_mod(mat: np.ndarray, mod: int) -> int:
    """Column rank over Z/mod via Gaussian elim (mod prime)."""
    m = np.array(mat, dtype=int) % mod
    rows, cols = m.shape
    r = 0
    for c in range(cols):
        piv = next((i for i in range(r, rows) if m[i, c] % mod), None)
        if piv is None:
            continue
        m[[r, piv]] = m[[piv, r]]
        inv = pow(int(m[r, c]) % mod, -1, mod)
        m[r] = (m[r] * inv) % mod
        for i in range(rows):
            if i != r and m[i, c] % mod:
                m[i] = (m[i] - m[i, c] * m[r]) % mod
        r += 1
        if r == rows:
            break
    return r


def koszul_h1(seq: list[int], mod: int) -> int:
    """dim ker d1 - rank d2 over Z/p (prime mod)."""
    d2, d1 = koszul_differentials(seq, mod)
    ker1 = 2 - z_rank_mod(d1, mod)
    im2 = z_rank_mod(d2, mod)
    return ker1 - im2


def _bench_koszul_complex(seed: int = 0) -> float:
    checks = []
    # regular sequence (2,3) over Z/5? elements must be units-mod-p for "regular" toy:
    # (1,2) in Z/5: x1 unit => H1 = 0 and H0 = R/(x) = 0
    checks.append(koszul_h1([1, 2], 5) == 0)
    # zero divisor pair (0,0) over Z/5: K = trivial complex, H1 = 2
    checks.append(koszul_h1([0, 0], 5) == 2)
    # (0,1): d1 = [0,1] rank1 ker dim1; d2 = [1;0] rank1 -> H1 = 0
    checks.append(koszul_h1([0, 1], 5) == 0)
    # differentials square to zero
    d2, d1 = koszul_differentials([2, 3], 5)
    checks.append(bool(np.all((d1 @ d2) % 5 == 0)))
    return float(sum(checks) / len(checks))


def bench_koszul_complex(seed: int = 0) -> dict[str, float]:
    return {"synthetic_koszul_complex": _bench_koszul_complex(seed)}
