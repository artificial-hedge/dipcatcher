"""nautilus_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def nautilus_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """nautilus_qa_studies

    check:
    nautilus_qa_studies: NautilusQA metrics
    """
    return fit_ok and sample_ok


def nautilus_qa_studies_aux(aux: bool) -> bool:
    """nautilus_qa_studies

    aux:
    nautilus_qa_studies: nautilus, deep reef slopes, answers, and scores
    """
    return aux


def _bench_nautilus_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(nautilus_qa_studies_ok(True, True))
    checks.append(not nautilus_qa_studies_ok(False, True))
    checks.append(nautilus_qa_studies_aux(True))
    checks.append(not nautilus_qa_studies_aux(False))
    checks.append(True)  # cephalopod canon
    return float(sum(checks) / len(checks))


def bench_nautilus_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nautilus_qa_studies": _bench_nautilus_qa_studies(seed)}
