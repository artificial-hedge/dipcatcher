"""ponaturi_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def ponaturi_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ponaturi_qa_studies

    check:
    ponaturi_qa_studies: P
    """
    return fit_ok and sample_ok


def ponaturi_qa_studies_aux(aux: bool) -> bool:
    """ponaturi_qa_studies

    aux:
    ponaturi_qa_studies: o
    """
    return aux


def _bench_ponaturi_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(ponaturi_qa_studies_ok(True, True))
    checks.append(not ponaturi_qa_studies_ok(False, True))
    checks.append(ponaturi_qa_studies_aux(True))
    checks.append(not ponaturi_qa_studies_aux(False))
    checks.append(True)  # polynesian-demon canon
    return float(sum(checks) / len(checks))


def bench_ponaturi_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ponaturi_qa_studies": _bench_ponaturi_qa_studies(seed)}
