"""book_sum_studies module (SYNTHETIC)."""

from __future__ import annotations


def book_sum_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """book_sum_studies

    check:
    book_sum_studies: BookSum metrics
    """
    return fit_ok and sample_ok


def book_sum_studies_aux(aux: bool) -> bool:
    """book_sum_studies

    aux:
    book_sum_studies: chapters, summaries, coherence, and scores
    """
    return aux


def _bench_book_sum_studies(seed: int = 0) -> float:
    checks = []
    checks.append(book_sum_studies_ok(True, True))
    checks.append(not book_sum_studies_ok(False, True))
    checks.append(book_sum_studies_aux(True))
    checks.append(not book_sum_studies_aux(False))
    checks.append(True)  # long-doc-sum canon
    return float(sum(checks) / len(checks))


def bench_book_sum_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_book_sum_studies": _bench_book_sum_studies(seed)}
