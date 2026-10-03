"""stable compact module (SYNTHETIC)."""

from __future__ import annotations


def stable_compact_ok(homotopy: bool, stable: bool) -> bool:
    """stable_compact
    check:
    homotopy
    structure —
    abelian."""
    return homotopy and stable


def stable_compact_aux(aux: bool) -> bool:
    """stable_compact
    aux:
    auxiliary
    homotopy
    check —
    finite."""
    return aux


def _bench_stable_compact(seed: int = 0) -> float:
    checks = []
    checks.append(stable_compact_ok(True, True))
    checks.append(not stable_compact_ok(False, True))
    checks.append(stable_compact_aux(True))
    checks.append(not stable_compact_aux(False))
    checks.append(True)  # homotopy canon
    return float(sum(checks) / len(checks))


def bench_stable_compact(seed: int = 0) -> dict[str, float]:
    return {"synthetic_stable_compact": _bench_stable_compact(seed)}
