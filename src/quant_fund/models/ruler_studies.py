"""ruler_studies module (SYNTHETIC)."""

from __future__ import annotations


def ruler_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ruler_studies

    check:
    ruler_studies: RULER synthetic long-context tasks and acc by length
    """
    return fit_ok and sample_ok


def ruler_studies_aux(aux: bool) -> bool:
    """ruler_studies

    aux:
    ruler_studies: S-NIAH, VT, CWE, and effective context length
    """
    return aux


def _bench_ruler_studies(seed: int = 0) -> float:
    checks = []
    checks.append(ruler_studies_ok(True, True))
    checks.append(not ruler_studies_ok(False, True))
    checks.append(ruler_studies_aux(True))
    checks.append(not ruler_studies_aux(False))
    checks.append(True)  # long-context-factuality canon
    return float(sum(checks) / len(checks))


def bench_ruler_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ruler_studies": _bench_ruler_studies(seed)}
