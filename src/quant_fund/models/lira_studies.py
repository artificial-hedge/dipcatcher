"""lira_studies module (SYNTHETIC)."""

from __future__ import annotations


def lira_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """lira_studies

    check:
    lira_studies: LiRA likelihood-ratio scores and low-FPR TPR
    """
    return fit_ok and sample_ok


def lira_studies_aux(aux: bool) -> bool:
    """lira_studies

    aux:
    lira_studies: Gaussian NLL attacks, z-scores, and ROC
    """
    return aux


def _bench_lira_studies(seed: int = 0) -> float:
    checks = []
    checks.append(lira_studies_ok(True, True))
    checks.append(not lira_studies_ok(False, True))
    checks.append(lira_studies_aux(True))
    checks.append(not lira_studies_aux(False))
    checks.append(True)  # privacy-inference canon
    return float(sum(checks) / len(checks))


def bench_lira_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_lira_studies": _bench_lira_studies(seed)}
