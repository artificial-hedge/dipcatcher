"""Uniform integrability of a finite family of distributions (SYNTHETIC)."""

from __future__ import annotations


def tail_mass(points: list[tuple[float, float]], m: float) -> float:
    """E[|X| 1_{|X| > M}] for a discrete rv given as (value, prob) pairs."""
    return sum(v * p for v, p in points if abs(v) > m)


def is_ui_family(fams: list[list[tuple[float, float]]], tol: float = 1e-9) -> bool:
    """The family is UI iff sup over members of tail_mass(M) -> 0 as
    M -> infty. On finite support families we probe M levels."""
    for m in (1e3, 1e6, 1e9):
        if max((tail_mass(f, m) for f in fams), default=0.0) > tol:
            return False
    return True


def _bench_uniform_integrability(seed: int = 0) -> float:
    checks = []
    # bounded family is UI
    fam1 = [[(1.0, 0.5), (-1.0, 0.5)], [(2.0, 0.5), (-2.0, 0.5)]]
    checks.append(is_ui_family(fam1))
    # family with unbounded support but decaying tails still UI on probe
    fam2 = [[(float(k), 2.0 ** (-k)) for k in range(1, 30)]]
    checks.append(is_ui_family(fam2))
    # non-UI: mass escaping to infinity with fixed tail mass
    fam3 = [[(float(10**j), 0.5), (0.0, 0.5)] for j in range(3, 10)]
    checks.append(not is_ui_family(fam3))
    # empty family vacuously UI
    checks.append(is_ui_family([]))
    # tail_mass computes correctly
    checks.append(abs(tail_mass([(4.0, 0.5), (1.0, 0.5)], 2.0) - 2.0) < 1e-12)
    checks.append(tail_mass([(4.0, 0.5), (1.0, 0.5)], 10.0) == 0.0)
    return float(sum(checks) / len(checks))


def bench_uniform_integrability(seed: int = 0) -> dict[str, float]:
    return {"synthetic_uniform_integrability": _bench_uniform_integrability(seed)}
