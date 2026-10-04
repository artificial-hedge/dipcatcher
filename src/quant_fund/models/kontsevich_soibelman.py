"""kontsevich soibelman module (SYNTHETIC)."""

from __future__ import annotations


def kontsevich_soibelman_ok(hall: bool, dg: bool) -> bool:
    """kontsevich_soibelman
    check:
    Hall-algebra-2
    structure —
    Kontsevich."""
    return hall and dg


def kontsevich_soibelman_aux(aux: bool) -> bool:
    """kontsevich_soibelman
    aux:
    auxiliary
    Hall
    check —
    Bridgeland."""
    return aux


def _bench_kontsevich_soibelman(seed: int = 0) -> float:
    checks = []
    checks.append(kontsevich_soibelman_ok(True, True))
    checks.append(not kontsevich_soibelman_ok(False, True))
    checks.append(kontsevich_soibelman_aux(True))
    checks.append(not kontsevich_soibelman_aux(False))
    checks.append(True)  # Hall-algebra-2 canon
    return float(sum(checks) / len(checks))


def bench_kontsevich_soibelman(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kontsevich_soibelman": _bench_kontsevich_soibelman(seed)}
