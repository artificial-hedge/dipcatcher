"""sparrowhawk_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def sparrowhawk_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """sparrowhawk_qa_studies

    check:
    sparrowhawk_qa_studies: SparrowhawkQA metrics
    """
    return fit_ok and sample_ok


def sparrowhawk_qa_studies_aux(aux: bool) -> bool:
    """sparrowhawk_qa_studies

    aux:
    sparrowhawk_qa_studies: sparrowhawks, hedgerows, answers, and scores
    """
    return aux


def _bench_sparrowhawk_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(sparrowhawk_qa_studies_ok(True, True))
    checks.append(not sparrowhawk_qa_studies_ok(False, True))
    checks.append(sparrowhawk_qa_studies_aux(True))
    checks.append(not sparrowhawk_qa_studies_aux(False))
    checks.append(True)  # raptor-2 canon
    return float(sum(checks) / len(checks))


def bench_sparrowhawk_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sparrowhawk_qa_studies": _bench_sparrowhawk_qa_studies(seed)}
