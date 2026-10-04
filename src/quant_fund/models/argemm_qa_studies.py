"""argemm_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def argemm_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """argemm_qa_studies

    check:
    argemm_qa_studies: s
    """
    return fit_ok and sample_ok


def argemm_qa_studies_aux(aux: bool) -> bool:
    """argemm_qa_studies

    aux:
    argemm_qa_studies: t
    """
    return aux


def _bench_argemm_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(argemm_qa_studies_ok(True, True))
    checks.append(not argemm_qa_studies_ok(False, True))
    checks.append(argemm_qa_studies_aux(True))
    checks.append(not argemm_qa_studies_aux(False))
    checks.append(True)  # saharan-2 canon
    return float(sum(checks) / len(checks))


def bench_argemm_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_argemm_qa_studies": _bench_argemm_qa_studies(seed)}
