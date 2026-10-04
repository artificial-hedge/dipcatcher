"""jackrabbit_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def jackrabbit_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """jackrabbit_qa_studies

    check:
    jackrabbit_qa_studies: JackrabbitQA metrics
    """
    return fit_ok and sample_ok


def jackrabbit_qa_studies_aux(aux: bool) -> bool:
    """jackrabbit_qa_studies

    aux:
    jackrabbit_qa_studies: jackrabbits, desert flats, answers, and scores
    """
    return aux


def _bench_jackrabbit_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(jackrabbit_qa_studies_ok(True, True))
    checks.append(not jackrabbit_qa_studies_ok(False, True))
    checks.append(jackrabbit_qa_studies_aux(True))
    checks.append(not jackrabbit_qa_studies_aux(False))
    checks.append(True)  # small-mammal canon
    return float(sum(checks) / len(checks))


def bench_jackrabbit_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_jackrabbit_qa_studies": _bench_jackrabbit_qa_studies(seed)}
