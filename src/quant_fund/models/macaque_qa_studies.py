"""macaque_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def macaque_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """macaque_qa_studies

    check:
    macaque_qa_studies: MacaqueQA metrics
    """
    return fit_ok and sample_ok


def macaque_qa_studies_aux(aux: bool) -> bool:
    """macaque_qa_studies

    aux:
    macaque_qa_studies: macaques, snowy troop, answers, and scores
    """
    return aux


def _bench_macaque_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(macaque_qa_studies_ok(True, True))
    checks.append(not macaque_qa_studies_ok(False, True))
    checks.append(macaque_qa_studies_aux(True))
    checks.append(not macaque_qa_studies_aux(False))
    checks.append(True)  # primate canon
    return float(sum(checks) / len(checks))


def bench_macaque_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_macaque_qa_studies": _bench_macaque_qa_studies(seed)}
