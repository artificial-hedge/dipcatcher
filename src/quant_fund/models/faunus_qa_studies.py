"""faunus_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def faunus_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """faunus_qa_studies

    check:
    faunus_qa_studies: FaunusQA metrics
    """
    return fit_ok and sample_ok


def faunus_qa_studies_aux(aux: bool) -> bool:
    """faunus_qa_studies

    aux:
    faunus_qa_studies: faunus, wild pipes, answers, and scores
    """
    return aux


def _bench_faunus_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(faunus_qa_studies_ok(True, True))
    checks.append(not faunus_qa_studies_ok(False, True))
    checks.append(faunus_qa_studies_aux(True))
    checks.append(not faunus_qa_studies_aux(False))
    checks.append(True)  # roman-minor-2 canon
    return float(sum(checks) / len(checks))


def bench_faunus_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_faunus_qa_studies": _bench_faunus_qa_studies(seed)}
