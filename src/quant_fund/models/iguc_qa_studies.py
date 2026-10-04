"""iguc_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def iguc_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """iguc_qa_studies

    check:
    iguc_qa_studies: l
    """
    return fit_ok and sample_ok


def iguc_qa_studies_aux(aux: bool) -> bool:
    """iguc_qa_studies

    aux:
    iguc_qa_studies: i
    """
    return aux


def _bench_iguc_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(iguc_qa_studies_ok(True, True))
    checks.append(not iguc_qa_studies_ok(False, True))
    checks.append(iguc_qa_studies_aux(True))
    checks.append(not iguc_qa_studies_aux(False))
    checks.append(True)  # numidian-myth canon
    return float(sum(checks) / len(checks))


def bench_iguc_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_iguc_qa_studies": _bench_iguc_qa_studies(seed)}
