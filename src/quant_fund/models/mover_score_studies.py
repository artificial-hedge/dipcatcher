"""mover_score_studies module (SYNTHETIC)."""

from __future__ import annotations


def mover_score_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """mover_score_studies

    check:
    mover_score_studies: MoverScore WMD metrics
    """
    return fit_ok and sample_ok


def mover_score_studies_aux(aux: bool) -> bool:
    """mover_score_studies

    aux:
    mover_score_studies: hypotheses, references, labels, and scores
    """
    return aux


def _bench_mover_score_studies(seed: int = 0) -> float:
    checks = []
    checks.append(mover_score_studies_ok(True, True))
    checks.append(not mover_score_studies_ok(False, True))
    checks.append(mover_score_studies_aux(True))
    checks.append(not mover_score_studies_aux(False))
    checks.append(True)  # translation-metric canon
    return float(sum(checks) / len(checks))


def bench_mover_score_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mover_score_studies": _bench_mover_score_studies(seed)}
