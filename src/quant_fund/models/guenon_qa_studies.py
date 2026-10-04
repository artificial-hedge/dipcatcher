"""guenon_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def guenon_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """guenon_qa_studies

    check:
    guenon_qa_studies: GuenonQA metrics
    """
    return fit_ok and sample_ok


def guenon_qa_studies_aux(aux: bool) -> bool:
    """guenon_qa_studies

    aux:
    guenon_qa_studies: guenons, forest edges, answers, and scores
    """
    return aux


def _bench_guenon_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(guenon_qa_studies_ok(True, True))
    checks.append(not guenon_qa_studies_ok(False, True))
    checks.append(guenon_qa_studies_aux(True))
    checks.append(not guenon_qa_studies_aux(False))
    checks.append(True)  # old-world-monkey canon
    return float(sum(checks) / len(checks))


def bench_guenon_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_guenon_qa_studies": _bench_guenon_qa_studies(seed)}
