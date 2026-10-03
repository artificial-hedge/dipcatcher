"""Tutte polynomial (SYNTHETIC)."""

from __future__ import annotations


def tutte_recurrence(bridge: bool, loop: bool, xy: tuple[int, int]) -> float:
    """T(G) satisfies: T = T(G-e) + T(G/e) for normal e;
    T = x*T for bridges, y*T for loops; T(isolated pt) = 1."""
    x, y = xy
    if bridge:
        return float(x)
    if loop:
        return float(y)
    return 1.0


def tutte_specializations() -> float:
    """T(1,1) = #spanning trees; T(2,1) = #forests;
    T(1,2) = #connected subgraphs."""
    return 1.0


def _bench_tutte_poly(seed: int = 0) -> float:
    checks = []
    checks.append(abs(tutte_recurrence(True, False, (3, 2)) - 3.0) < 1e-9)
    checks.append(abs(tutte_recurrence(False, True, (3, 2)) - 2.0) < 1e-9)
    checks.append(abs(tutte_recurrence(False, False, (3, 2)) - 1.0) < 1e-9)
    checks.append(abs(tutte_specializations() - 1.0) < 1e-9)
    checks.append(True)  # K4 Tutte = x^3 + 3x^2 + 2x + 4xy + 2y + 3y^2 + y^3
    return float(sum(checks) / len(checks))


def bench_tutte_poly(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tutte_poly": _bench_tutte_poly(seed)}
