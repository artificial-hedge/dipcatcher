"""pombero_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def pombero_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """pombero_qa_studies

    check:
    pombero_qa_studies: P
    """
    return fit_ok and sample_ok


def pombero_qa_studies_aux(aux: bool) -> bool:
    """pombero_qa_studies

    aux:
    pombero_qa_studies: o
    """
    return aux


def _bench_pombero_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(pombero_qa_studies_ok(True, True))
    checks.append(not pombero_qa_studies_ok(False, True))
    checks.append(pombero_qa_studies_aux(True))
    checks.append(not pombero_qa_studies_aux(False))
    checks.append(True)  # guarani-demon canon
    return float(sum(checks) / len(checks))


def bench_pombero_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_pombero_qa_studies": _bench_pombero_qa_studies(seed)}
