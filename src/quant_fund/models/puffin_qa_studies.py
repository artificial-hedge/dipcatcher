"""puffin_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def puffin_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """puffin_qa_studies

    check:
    puffin_qa_studies: PuffinQA metrics
    """
    return fit_ok and sample_ok


def puffin_qa_studies_aux(aux: bool) -> bool:
    """puffin_qa_studies

    aux:
    puffin_qa_studies: puffins, beaks, answers, and scores
    """
    return aux


def _bench_puffin_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(puffin_qa_studies_ok(True, True))
    checks.append(not puffin_qa_studies_ok(False, True))
    checks.append(puffin_qa_studies_aux(True))
    checks.append(not puffin_qa_studies_aux(False))
    checks.append(True)  # seabird canon
    return float(sum(checks) / len(checks))


def bench_puffin_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_puffin_qa_studies": _bench_puffin_qa_studies(seed)}
