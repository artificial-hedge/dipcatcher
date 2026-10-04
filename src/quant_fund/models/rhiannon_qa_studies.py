"""rhiannon_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def rhiannon_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """rhiannon_qa_studies

    check:
    rhiannon_qa_studies: RhiannonQA metrics
    """
    return fit_ok and sample_ok


def rhiannon_qa_studies_aux(aux: bool) -> bool:
    """rhiannon_qa_studies

    aux:
    rhiannon_qa_studies: rhiannon, mare riders, answers, and scores
    """
    return aux


def _bench_rhiannon_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(rhiannon_qa_studies_ok(True, True))
    checks.append(not rhiannon_qa_studies_ok(False, True))
    checks.append(rhiannon_qa_studies_aux(True))
    checks.append(not rhiannon_qa_studies_aux(False))
    checks.append(True)  # celtic-myth-3 canon
    return float(sum(checks) / len(checks))


def bench_rhiannon_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_rhiannon_qa_studies": _bench_rhiannon_qa_studies(seed)}
