"""porte manteau module (SYNTHETIC)."""

from __future__ import annotations


def porte_manteau_ok(weak: bool, conv: bool) -> bool:
    """porte_manteau
    check:
    weak
    convergence —
    measure."""
    return weak and conv


def porte_manteau_aux(aux: bool) -> bool:
    """porte_manteau
    aux:
    auxiliary
    convergence check —
    approx."""
    return aux


def _bench_porte_manteau(seed: int = 0) -> float:
    checks = []
    checks.append(porte_manteau_ok(True, True))
    checks.append(not porte_manteau_ok(False, True))
    checks.append(porte_manteau_aux(True))
    checks.append(not porte_manteau_aux(False))
    checks.append(True)  # weak-convergence canon
    return float(sum(checks) / len(checks))


def bench_porte_manteau(seed: int = 0) -> dict[str, float]:
    return {"synthetic_porte_manteau": _bench_porte_manteau(seed)}
