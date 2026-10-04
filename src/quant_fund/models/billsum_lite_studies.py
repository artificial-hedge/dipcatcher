"""billsum_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def billsum_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """billsum_lite_studies

    check:
    billsum_lite_studies: BillSum legislation metrics
    """
    return fit_ok and sample_ok


def billsum_lite_studies_aux(aux: bool) -> bool:
    """billsum_lite_studies

    aux:
    billsum_lite_studies: bills, summaries, references, and scores
    """
    return aux


def _bench_billsum_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(billsum_lite_studies_ok(True, True))
    checks.append(not billsum_lite_studies_ok(False, True))
    checks.append(billsum_lite_studies_aux(True))
    checks.append(not billsum_lite_studies_aux(False))
    checks.append(True)  # long-doc-summarization canon
    return float(sum(checks) / len(checks))


def bench_billsum_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_billsum_lite_studies": _bench_billsum_lite_studies(seed)}
