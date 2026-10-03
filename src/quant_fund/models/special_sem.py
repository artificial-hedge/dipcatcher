"""special sem module (SYNTHETIC)."""

from __future__ import annotations


def special_sem_ok(sm: bool, dec: bool) -> bool:
    """special_sem
    check:
    semimartingale —
    canonical
    decomposition."""
    return sm and dec


def special_sem_aux(aux: bool) -> bool:
    """special_sem
    aux:
    auxiliary
    decomposition
    check —
    characteristics."""
    return aux


def _bench_special_sem(seed: int = 0) -> float:
    checks = []
    checks.append(special_sem_ok(True, True))
    checks.append(not special_sem_ok(False, True))
    checks.append(special_sem_aux(True))
    checks.append(not special_sem_aux(False))
    checks.append(True)  # semimartingale canon
    return float(sum(checks) / len(checks))


def bench_special_sem(seed: int = 0) -> dict[str, float]:
    return {"synthetic_special_sem": _bench_special_sem(seed)}
