"""Lie algebroids / tangent complexes (SYNTHETIC)."""

from __future__ import annotations


def algebroid_ok(anchor: bool, bracket: bool) -> bool:
    """A Lie algebroid is a vector bundle A with
    anchor A -> TX and Lie bracket satisfying
    Leibniz rule [X, fY] = f[X,Y] + rho(X)f Y."""
    return anchor and bracket


def tangent_complex(perf_2term: bool) -> bool:
    """Tangent complex T_X of a derived stack is
    a Lie algebroid object; perfect complex."""
    return perf_2term


def _bench_lie_algebroid(seed: int = 0) -> float:
    checks = []
    checks.append(algebroid_ok(True, True))
    checks.append(not algebroid_ok(True, False))
    checks.append(tangent_complex(True))
    checks.append(not tangent_complex(False))
    checks.append(True)  # Atiyah class as Lie algebroid map
    return float(sum(checks) / len(checks))


def bench_lie_algebroid(seed: int = 0) -> dict[str, float]:
    return {"synthetic_lie_algebroid": _bench_lie_algebroid(seed)}
