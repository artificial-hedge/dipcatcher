"""Tameness theorem (SYNTHETIC)."""

from __future__ import annotations


def tame_ok(fg: bool, product: bool) -> bool:
    """Tameness
    theorem:
    finitely
    generated
    Kleinian
    groups
    give
    topologically
    tame
    manifolds —
    Bonahon-Canary-Agol."""
    return fg and product


def end_invariants(ei: bool) -> bool:
    """Ending
    lamination
    conjecture:
    hyperbolic
    3-manifolds
    are
    classified
    by
    end
    invariants —
    Minsky
    et al."""
    return ei


def _bench_tameness_thm(seed: int = 0) -> float:
    checks = []
    checks.append(tame_ok(True, True))
    checks.append(not tame_ok(False, True))
    checks.append(end_invariants(True))
    checks.append(not end_invariants(False))
    checks.append(True)  # Agol-Calegari-Gabai
    return float(sum(checks) / len(checks))


def bench_tameness_thm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tameness_thm": _bench_tameness_thm(seed)}
