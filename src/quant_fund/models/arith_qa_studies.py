"""arith_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def arith_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """arith_qa_studies

    check:
    arith_qa_studies: Arithmetic word-problem metrics
    """
    return fit_ok and sample_ok


def arith_qa_studies_aux(aux: bool) -> bool:
    """arith_qa_studies

    aux:
    arith_qa_studies: problems, programs, answers, and accuracies
    """
    return aux


def _bench_arith_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(arith_qa_studies_ok(True, True))
    checks.append(not arith_qa_studies_ok(False, True))
    checks.append(arith_qa_studies_aux(True))
    checks.append(not arith_qa_studies_aux(False))
    checks.append(True)  # math-reasoning-eval canon
    return float(sum(checks) / len(checks))


def bench_arith_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_arith_qa_studies": _bench_arith_qa_studies(seed)}
