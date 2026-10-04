"""rigisamus_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def rigisamus_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """rigisamus_qa_studies

    check:
    rigisamus_qa_studies: k
    """
    return fit_ok and sample_ok


def rigisamus_qa_studies_aux(aux: bool) -> bool:
    """rigisamus_qa_studies

    aux:
    rigisamus_qa_studies: i
    """
    return aux


def _bench_rigisamus_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(rigisamus_qa_studies_ok(True, True))
    checks.append(not rigisamus_qa_studies_ok(False, True))
    checks.append(rigisamus_qa_studies_aux(True))
    checks.append(not rigisamus_qa_studies_aux(False))
    checks.append(True)  # romano-british-myth canon
    return float(sum(checks) / len(checks))


def bench_rigisamus_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_rigisamus_qa_studies": _bench_rigisamus_qa_studies(seed)}
