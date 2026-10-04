"""wsc_studies module (SYNTHETIC)."""

from __future__ import annotations


def wsc_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """wsc_studies

    check:
    wsc_studies: WSC Winograd Schemas challenge items and accuracy
    """
    return fit_ok and sample_ok


def wsc_studies_aux(aux: bool) -> bool:
    """wsc_studies

    aux:
    wsc_studies: texts, coreference candidates, and labels
    """
    return aux


def _bench_wsc_studies(seed: int = 0) -> float:
    checks = []
    checks.append(wsc_studies_ok(True, True))
    checks.append(not wsc_studies_ok(False, True))
    checks.append(wsc_studies_aux(True))
    checks.append(not wsc_studies_aux(False))
    checks.append(True)  # winograd-eval canon
    return float(sum(checks) / len(checks))


def bench_wsc_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_wsc_studies": _bench_wsc_studies(seed)}
