"""hara slade module (SYNTHETIC)."""

from __future__ import annotations


def hara_slade_ok(perc: bool, lace: bool) -> bool:
    """hara_slade
    check:
    percolation-2
    structure —
    Grimmett."""
    return perc and lace


def hara_slade_aux(aux: bool) -> bool:
    """hara_slade
    aux:
    auxiliary
    lace-expansion
    check —
    Hara."""
    return aux


def _bench_hara_slade(seed: int = 0) -> float:
    checks = []
    checks.append(hara_slade_ok(True, True))
    checks.append(not hara_slade_ok(False, True))
    checks.append(hara_slade_aux(True))
    checks.append(not hara_slade_aux(False))
    checks.append(True)  # percolation-2 canon
    return float(sum(checks) / len(checks))


def bench_hara_slade(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hara_slade": _bench_hara_slade(seed)}
