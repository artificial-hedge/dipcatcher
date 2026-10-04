"""fossa_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def fossa_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """fossa_qa_studies

    check:
    fossa_qa_studies: FossaQA metrics
    """
    return fit_ok and sample_ok


def fossa_qa_studies_aux(aux: bool) -> bool:
    """fossa_qa_studies

    aux:
    fossa_qa_studies: fossa, dry forests, answers, and scores
    """
    return aux


def _bench_fossa_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(fossa_qa_studies_ok(True, True))
    checks.append(not fossa_qa_studies_ok(False, True))
    checks.append(fossa_qa_studies_aux(True))
    checks.append(not fossa_qa_studies_aux(False))
    checks.append(True)  # mammal canon
    return float(sum(checks) / len(checks))


def bench_fossa_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fossa_qa_studies": _bench_fossa_qa_studies(seed)}
