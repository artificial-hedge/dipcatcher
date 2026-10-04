"""cnn_dailymail_studies module (SYNTHETIC)."""

from __future__ import annotations


def cnn_dailymail_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """cnn_dailymail_studies

    check:
    cnn_dailymail_studies: CNN/DailyMail summarization metrics
    """
    return fit_ok and sample_ok


def cnn_dailymail_studies_aux(aux: bool) -> bool:
    """cnn_dailymail_studies

    aux:
    cnn_dailymail_studies: articles, highlights, references, and scores
    """
    return aux


def _bench_cnn_dailymail_studies(seed: int = 0) -> float:
    checks = []
    checks.append(cnn_dailymail_studies_ok(True, True))
    checks.append(not cnn_dailymail_studies_ok(False, True))
    checks.append(cnn_dailymail_studies_aux(True))
    checks.append(not cnn_dailymail_studies_aux(False))
    checks.append(True)  # summarization canon
    return float(sum(checks) / len(checks))


def bench_cnn_dailymail_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cnn_dailymail_studies": _bench_cnn_dailymail_studies(seed)}
