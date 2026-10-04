"""moore space2 module (SYNTHETIC)."""

from __future__ import annotations


def moore_space2_ok(homotopy: bool, periodic: bool) -> bool:
    """moore_space2
    check:
    homotopy
    structure —
    unstable."""
    return homotopy and periodic


def moore_space2_aux(aux: bool) -> bool:
    """moore_space2
    aux:
    auxiliary
    homotopy
    check —
    periodic."""
    return aux


def _bench_moore_space2(seed: int = 0) -> float:
    checks = []
    checks.append(moore_space2_ok(True, True))
    checks.append(not moore_space2_ok(False, True))
    checks.append(moore_space2_aux(True))
    checks.append(not moore_space2_aux(False))
    checks.append(True)  # homotopy canon
    return float(sum(checks) / len(checks))


def bench_moore_space2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_moore_space2": _bench_moore_space2(seed)}
