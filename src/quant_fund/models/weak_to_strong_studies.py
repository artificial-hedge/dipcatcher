"""weak_to_strong_studies module (SYNTHETIC)."""

from __future__ import annotations


def weak_to_strong_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """weak_to_strong_studies

    check:
    weak_to_strong_studies: weak supervision generalization/students and teachers
    """
    return fit_ok and sample_ok


def weak_to_strong_studies_aux(aux: bool) -> bool:
    """weak_to_strong_studies

    aux:
    weak_to_strong_studies: auxiliary confidence and elicitation/recovery and ceiling
    """
    return aux


def _bench_weak_to_strong_studies(seed: int = 0) -> float:
    checks = []
    checks.append(weak_to_strong_studies_ok(True, True))
    checks.append(not weak_to_strong_studies_ok(False, True))
    checks.append(weak_to_strong_studies_aux(True))
    checks.append(not weak_to_strong_studies_aux(False))
    checks.append(True)  # scalable-oversight canon
    return float(sum(checks) / len(checks))


def bench_weak_to_strong_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_weak_to_strong_studies": _bench_weak_to_strong_studies(seed)}
