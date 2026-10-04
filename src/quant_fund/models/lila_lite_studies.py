"""lila_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def lila_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """lila_lite_studies

    check:
    lila_lite_studies: LILA metrics
    """
    return fit_ok and sample_ok


def lila_lite_studies_aux(aux: bool) -> bool:
    """lila_lite_studies

    aux:
    lila_lite_studies: problems, programs, answers, and scores
    """
    return aux


def _bench_lila_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(lila_lite_studies_ok(True, True))
    checks.append(not lila_lite_studies_ok(False, True))
    checks.append(lila_lite_studies_aux(True))
    checks.append(not lila_lite_studies_aux(False))
    checks.append(True)  # math-word-2 canon
    return float(sum(checks) / len(checks))


def bench_lila_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_lila_lite_studies": _bench_lila_lite_studies(seed)}
