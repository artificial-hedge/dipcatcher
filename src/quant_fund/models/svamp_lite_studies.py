"""svamp_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def svamp_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """svamp_lite_studies

    check:
    svamp_lite_studies: SVAMP arithmetic metrics
    """
    return fit_ok and sample_ok


def svamp_lite_studies_aux(aux: bool) -> bool:
    """svamp_lite_studies

    aux:
    svamp_lite_studies: problems, equations, answers, and scores
    """
    return aux


def _bench_svamp_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(svamp_lite_studies_ok(True, True))
    checks.append(not svamp_lite_studies_ok(False, True))
    checks.append(svamp_lite_studies_aux(True))
    checks.append(not svamp_lite_studies_aux(False))
    checks.append(True)  # math-word-problem canon
    return float(sum(checks) / len(checks))


def bench_svamp_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_svamp_lite_studies": _bench_svamp_lite_studies(seed)}
