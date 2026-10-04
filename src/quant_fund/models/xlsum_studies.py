"""xlsum_studies module (SYNTHETIC)."""

from __future__ import annotations


def xlsum_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """xlsum_studies

    check:
    xlsum_studies: XL-Sum multilingual summarization metrics
    """
    return fit_ok and sample_ok


def xlsum_studies_aux(aux: bool) -> bool:
    """xlsum_studies

    aux:
    xlsum_studies: articles, summaries, languages, and scores
    """
    return aux


def _bench_xlsum_studies(seed: int = 0) -> float:
    checks = []
    checks.append(xlsum_studies_ok(True, True))
    checks.append(not xlsum_studies_ok(False, True))
    checks.append(xlsum_studies_aux(True))
    checks.append(not xlsum_studies_aux(False))
    checks.append(True)  # multilingual-eval canon
    return float(sum(checks) / len(checks))


def bench_xlsum_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_xlsum_studies": _bench_xlsum_studies(seed)}
