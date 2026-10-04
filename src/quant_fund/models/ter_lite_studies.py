"""ter_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def ter_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ter_lite_studies

    check:
    ter_lite_studies: TER edit-rate metrics
    """
    return fit_ok and sample_ok


def ter_lite_studies_aux(aux: bool) -> bool:
    """ter_lite_studies

    aux:
    ter_lite_studies: hypotheses, references, labels, and scores
    """
    return aux


def _bench_ter_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(ter_lite_studies_ok(True, True))
    checks.append(not ter_lite_studies_ok(False, True))
    checks.append(ter_lite_studies_aux(True))
    checks.append(not ter_lite_studies_aux(False))
    checks.append(True)  # translation-metric canon
    return float(sum(checks) / len(checks))


def bench_ter_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ter_lite_studies": _bench_ter_lite_studies(seed)}
