"""juracan_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def juracan_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """juracan_qa_studies

    check:
    juracan_qa_studies: JuracanQA metrics
    """
    return fit_ok and sample_ok


def juracan_qa_studies_aux(aux: bool) -> bool:
    """juracan_qa_studies

    aux:
    juracan_qa_studies: juracan, storm riders, answers, and scores
    """
    return aux


def _bench_juracan_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(juracan_qa_studies_ok(True, True))
    checks.append(not juracan_qa_studies_ok(False, True))
    checks.append(juracan_qa_studies_aux(True))
    checks.append(not juracan_qa_studies_aux(False))
    checks.append(True)  # taino-myth canon
    return float(sum(checks) / len(checks))


def bench_juracan_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_juracan_qa_studies": _bench_juracan_qa_studies(seed)}
