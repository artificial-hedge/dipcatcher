"""barnacle_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def barnacle_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """barnacle_qa_studies

    check:
    barnacle_qa_studies: BarnacleQA metrics
    """
    return fit_ok and sample_ok


def barnacle_qa_studies_aux(aux: bool) -> bool:
    """barnacle_qa_studies

    aux:
    barnacle_qa_studies: barnacles, rock shelves, answers, and scores
    """
    return aux


def _bench_barnacle_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(barnacle_qa_studies_ok(True, True))
    checks.append(not barnacle_qa_studies_ok(False, True))
    checks.append(barnacle_qa_studies_aux(True))
    checks.append(not barnacle_qa_studies_aux(False))
    checks.append(True)  # plankton-shore canon
    return float(sum(checks) / len(checks))


def bench_barnacle_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_barnacle_qa_studies": _bench_barnacle_qa_studies(seed)}
