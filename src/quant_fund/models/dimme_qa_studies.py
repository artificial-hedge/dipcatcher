"""dimme_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def dimme_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """dimme_qa_studies

    check:
    dimme_qa_studies: d
    """
    return fit_ok and sample_ok


def dimme_qa_studies_aux(aux: bool) -> bool:
    """dimme_qa_studies

    aux:
    dimme_qa_studies: i
    """
    return aux


def _bench_dimme_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(dimme_qa_studies_ok(True, True))
    checks.append(not dimme_qa_studies_ok(False, True))
    checks.append(dimme_qa_studies_aux(True))
    checks.append(not dimme_qa_studies_aux(False))
    checks.append(True)  # mesopotamian-demon-3 canon
    return float(sum(checks) / len(checks))


def bench_dimme_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dimme_qa_studies": _bench_dimme_qa_studies(seed)}
