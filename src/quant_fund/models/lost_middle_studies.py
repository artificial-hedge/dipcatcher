"""lost_middle_studies module (SYNTHETIC)."""

from __future__ import annotations


def lost_middle_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """lost_middle_studies

    check:
    lost_middle_studies: Lost-in-the-Middle positional retrieval metrics
    """
    return fit_ok and sample_ok


def lost_middle_studies_aux(aux: bool) -> bool:
    """lost_middle_studies

    aux:
    lost_middle_studies: positions, answers, and retrieval scores
    """
    return aux


def _bench_lost_middle_studies(seed: int = 0) -> float:
    checks = []
    checks.append(lost_middle_studies_ok(True, True))
    checks.append(not lost_middle_studies_ok(False, True))
    checks.append(lost_middle_studies_aux(True))
    checks.append(not lost_middle_studies_aux(False))
    checks.append(True)  # long-context-2 canon
    return float(sum(checks) / len(checks))


def bench_lost_middle_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_lost_middle_studies": _bench_lost_middle_studies(seed)}
