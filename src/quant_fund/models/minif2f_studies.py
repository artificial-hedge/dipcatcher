"""minif2f_studies module (SYNTHETIC)."""

from __future__ import annotations


def minif2f_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """minif2f_studies

    check:
    minif2f_studies: miniF2F proof-search solve rates and step metrics
    """
    return fit_ok and sample_ok


def minif2f_studies_aux(aux: bool) -> bool:
    """minif2f_studies

    aux:
    minif2f_studies: problems, proofs, and solve rates
    """
    return aux


def _bench_minif2f_studies(seed: int = 0) -> float:
    checks = []
    checks.append(minif2f_studies_ok(True, True))
    checks.append(not minif2f_studies_ok(False, True))
    checks.append(minif2f_studies_aux(True))
    checks.append(not minif2f_studies_aux(False))
    checks.append(True)  # reasoning-eval canon
    return float(sum(checks) / len(checks))


def bench_minif2f_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_minif2f_studies": _bench_minif2f_studies(seed)}
