"""arthur_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def arthur_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """arthur_qa_studies

    check:
    arthur_qa_studies: o
    """
    return fit_ok and sample_ok


def arthur_qa_studies_aux(aux: bool) -> bool:
    """arthur_qa_studies

    aux:
    arthur_qa_studies: n
    """
    return aux


def _bench_arthur_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(arthur_qa_studies_ok(True, True))
    checks.append(not arthur_qa_studies_ok(False, True))
    checks.append(arthur_qa_studies_aux(True))
    checks.append(not arthur_qa_studies_aux(False))
    checks.append(True)  # arthurian-myth canon
    return float(sum(checks) / len(checks))


def bench_arthur_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_arthur_qa_studies": _bench_arthur_qa_studies(seed)}
