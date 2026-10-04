"""booksum_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def booksum_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """booksum_lite_studies

    check:
    booksum_lite_studies: BookSum long-form metrics
    """
    return fit_ok and sample_ok


def booksum_lite_studies_aux(aux: bool) -> bool:
    """booksum_lite_studies

    aux:
    booksum_lite_studies: chapters, summaries, references, and scores
    """
    return aux


def _bench_booksum_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(booksum_lite_studies_ok(True, True))
    checks.append(not booksum_lite_studies_ok(False, True))
    checks.append(booksum_lite_studies_aux(True))
    checks.append(not booksum_lite_studies_aux(False))
    checks.append(True)  # long-doc-summarization canon
    return float(sum(checks) / len(checks))


def bench_booksum_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_booksum_lite_studies": _bench_booksum_lite_studies(seed)}
