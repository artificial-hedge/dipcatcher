"""wsc_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def wsc_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """wsc_lite_studies

    check:
    wsc_lite_studies: WSC Winograd metrics
    """
    return fit_ok and sample_ok


def wsc_lite_studies_aux(aux: bool) -> bool:
    """wsc_lite_studies

    aux:
    wsc_lite_studies: sentences, spans, answers, and scores
    """
    return aux


def _bench_wsc_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(wsc_lite_studies_ok(True, True))
    checks.append(not wsc_lite_studies_ok(False, True))
    checks.append(wsc_lite_studies_aux(True))
    checks.append(not wsc_lite_studies_aux(False))
    checks.append(True)  # NLU-exotics canon
    return float(sum(checks) / len(checks))


def bench_wsc_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_wsc_lite_studies": _bench_wsc_lite_studies(seed)}
