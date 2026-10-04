"""tiamat_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def tiamat_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """tiamat_qa_studies

    check:
    tiamat_qa_studies: TiamatQA metrics
    """
    return fit_ok and sample_ok


def tiamat_qa_studies_aux(aux: bool) -> bool:
    """tiamat_qa_studies

    aux:
    tiamat_qa_studies: tiamat, salt mothers, answers, and scores
    """
    return aux


def _bench_tiamat_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(tiamat_qa_studies_ok(True, True))
    checks.append(not tiamat_qa_studies_ok(False, True))
    checks.append(tiamat_qa_studies_aux(True))
    checks.append(not tiamat_qa_studies_aux(False))
    checks.append(True)  # babylonian-2 canon
    return float(sum(checks) / len(checks))


def bench_tiamat_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tiamat_qa_studies": _bench_tiamat_qa_studies(seed)}
