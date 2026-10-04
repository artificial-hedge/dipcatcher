"""ninlil_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def ninlil_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ninlil_qa_studies

    check:
    ninlil_qa_studies: NinlilQA metrics
    """
    return fit_ok and sample_ok


def ninlil_qa_studies_aux(aux: bool) -> bool:
    """ninlil_qa_studies

    aux:
    ninlil_qa_studies: ninlil, grain mothers, answers, and scores
    """
    return aux


def _bench_ninlil_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(ninlil_qa_studies_ok(True, True))
    checks.append(not ninlil_qa_studies_ok(False, True))
    checks.append(ninlil_qa_studies_aux(True))
    checks.append(not ninlil_qa_studies_aux(False))
    checks.append(True)  # babylonian-2 canon
    return float(sum(checks) / len(checks))


def bench_ninlil_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ninlil_qa_studies": _bench_ninlil_qa_studies(seed)}
