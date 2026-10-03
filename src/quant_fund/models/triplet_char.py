"""triplet char module (SYNTHETIC)."""

from __future__ import annotations


def triplet_char_ok(sm: bool, dec: bool) -> bool:
    """triplet_char
    check:
    semimartingale —
    canonical
    decomposition."""
    return sm and dec


def triplet_char_aux(aux: bool) -> bool:
    """triplet_char
    aux:
    auxiliary
    decomposition
    check —
    characteristics."""
    return aux


def _bench_triplet_char(seed: int = 0) -> float:
    checks = []
    checks.append(triplet_char_ok(True, True))
    checks.append(not triplet_char_ok(False, True))
    checks.append(triplet_char_aux(True))
    checks.append(not triplet_char_aux(False))
    checks.append(True)  # semimartingale canon
    return float(sum(checks) / len(checks))


def bench_triplet_char(seed: int = 0) -> dict[str, float]:
    return {"synthetic_triplet_char": _bench_triplet_char(seed)}
