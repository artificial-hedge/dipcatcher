"""Cyclotomic polynomials Phi_n: product_{d|n} Phi_d = x^n - 1 (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def cyclotomic(n: int) -> list[float]:
    """Phi_n via Moebius recursion from x^n - 1 = prod_{d|n} Phi_d."""
    divisors = sorted(d for d in range(1, n + 1) if n % d == 0)
    phis: dict[int, list[float]] = {}
    for d in divisors:
        # Phi_d = (x^d - 1) / prod_{e|d, e<d} Phi_e
        num = [0.0] * (d + 1)
        num[0], num[d] = -1.0, 1.0
        for e in divisors:
            if e < d and d % e == 0:
                num = _poly_div_exact(num, phis[e])
        phis[d] = num
    return phis[n]


def _poly_div_exact(f: list[float], g: list[float]) -> list[float]:
    r = list(f)
    q = [0.0] * (len(f) - len(g) + 1)
    for i in range(len(q) - 1, -1, -1):
        c = r[len(g) - 1 + i] / g[-1]
        q[i] = c
        for j, gv in enumerate(g):
            r[j + i] -= c * gv
    return q


def _bench_cyclotomic_poly(seed: int = 0) -> float:
    checks = []
    # Phi_3 = x^2 + x + 1
    checks.append(np.allclose(cyclotomic(3), [1.0, 1.0, 1.0]))
    # Phi_4 = x^2 + 1
    checks.append(np.allclose(cyclotomic(4), [1.0, 0.0, 1.0]))
    # Phi_6 = x^2 - x + 1
    checks.append(np.allclose(cyclotomic(6), [1.0, -1.0, 1.0]))
    # Phi_5 = x^4+x^3+x^2+x+1
    checks.append(np.allclose(cyclotomic(5), [1.0, 1.0, 1.0, 1.0, 1.0]))
    # product_{d|6} Phi_d = x^6 - 1
    prod = np.polynomial.polynomial.polymul(
        np.polynomial.polynomial.polymul(cyclotomic(1), cyclotomic(2)),
        np.polynomial.polynomial.polymul(cyclotomic(3), cyclotomic(6)),
    )
    checks.append(bool(np.allclose(prod, [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])))
    # primitive n-th roots are roots of Phi_n: Phi_6(e^{i pi/3}) = 0
    z = np.exp(1j * np.pi / 3)
    checks.append(abs(complex(np.polyval(cyclotomic(6)[::-1], z))) < 1e-9)
    # deg Phi_n = phi(n)
    checks.append(len(cyclotomic(8)) - 1 == 4)  # phi(8) = 4
    return float(sum(checks) / len(checks))


def bench_cyclotomic_poly(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cyclotomic_poly": _bench_cyclotomic_poly(seed)}
