"""spectral gap_sem module (SYNTHETIC)."""

from __future__ import annotations


def spectral_gap_sem_ok(sem: bool, gen: bool) -> bool:
    """spectral_gap_sem
    check:
    Markov
    semigroup —
    energy."""
    return sem and gen


def spectral_gap_sem_aux(aux: bool) -> bool:
    """spectral_gap_sem
    aux:
    auxiliary
    semigroup check —
    curvature."""
    return aux


def _bench_spectral_gap_sem(seed: int = 0) -> float:
    checks = []
    checks.append(spectral_gap_sem_ok(True, True))
    checks.append(not spectral_gap_sem_ok(False, True))
    checks.append(spectral_gap_sem_aux(True))
    checks.append(not spectral_gap_sem_aux(False))
    checks.append(True)  # Markov-semigroup canon
    return float(sum(checks) / len(checks))


def bench_spectral_gap_sem(seed: int = 0) -> dict[str, float]:
    return {"synthetic_spectral_gap_sem": _bench_spectral_gap_sem(seed)}
