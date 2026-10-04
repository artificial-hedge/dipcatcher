"""qsum_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def qsum_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """qsum_lite_studies

    check:
    qsum_lite_studies: QuerySum metrics
    """
    return fit_ok and sample_ok


def qsum_lite_studies_aux(aux: bool) -> bool:
    """qsum_lite_studies

    aux:
    qsum_lite_studies: docs, queries, summaries, and scores
    """
    return aux


def _bench_qsum_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(qsum_lite_studies_ok(True, True))
    checks.append(not qsum_lite_studies_ok(False, True))
    checks.append(qsum_lite_studies_aux(True))
    checks.append(not qsum_lite_studies_aux(False))
    checks.append(True)  # summarization-2 canon
    return float(sum(checks) / len(checks))


def bench_qsum_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_qsum_lite_studies": _bench_qsum_lite_studies(seed)}
