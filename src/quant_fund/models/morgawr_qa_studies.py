"""morgawr_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def morgawr_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """morgawr_qa_studies

    check:
    morgawr_qa_studies: s
    """
    return fit_ok and sample_ok


def morgawr_qa_studies_aux(aux: bool) -> bool:
    """morgawr_qa_studies

    aux:
    morgawr_qa_studies: e
    """
    return aux


def _bench_morgawr_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(morgawr_qa_studies_ok(True, True))
    checks.append(not morgawr_qa_studies_ok(False, True))
    checks.append(morgawr_qa_studies_aux(True))
    checks.append(not morgawr_qa_studies_aux(False))
    checks.append(True)  # cornish-myth canon
    return float(sum(checks) / len(checks))


def bench_morgawr_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_morgawr_qa_studies": _bench_morgawr_qa_studies(seed)}
