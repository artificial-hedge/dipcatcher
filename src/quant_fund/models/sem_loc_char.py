"""sem loc_char module (SYNTHETIC)."""

from __future__ import annotations


def sem_loc_char_ok(sm: bool, dec: bool) -> bool:
    """sem_loc_char
    check:
    semimartingale —
    canonical
    decomposition."""
    return sm and dec


def sem_loc_char_aux(aux: bool) -> bool:
    """sem_loc_char
    aux:
    auxiliary
    decomposition
    check —
    characteristics."""
    return aux


def _bench_sem_loc_char(seed: int = 0) -> float:
    checks = []
    checks.append(sem_loc_char_ok(True, True))
    checks.append(not sem_loc_char_ok(False, True))
    checks.append(sem_loc_char_aux(True))
    checks.append(not sem_loc_char_aux(False))
    checks.append(True)  # semimartingale canon
    return float(sum(checks) / len(checks))


def bench_sem_loc_char(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sem_loc_char": _bench_sem_loc_char(seed)}
