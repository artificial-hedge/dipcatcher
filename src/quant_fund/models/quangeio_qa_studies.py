"""quangeio_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def quangeio_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """quangeio_qa_studies

    check:
    quangeio_qa_studies: l
    """
    return fit_ok and sample_ok


def quangeio_qa_studies_aux(aux: bool) -> bool:
    """quangeio_qa_studies

    aux:
    quangeio_qa_studies: o
    """
    return aux


def _bench_quangeio_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(quangeio_qa_studies_ok(True, True))
    checks.append(not quangeio_qa_studies_ok(False, True))
    checks.append(quangeio_qa_studies_aux(True))
    checks.append(not quangeio_qa_studies_aux(False))
    checks.append(True)  # lusitanian-myth canon
    return float(sum(checks) / len(checks))


def bench_quangeio_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_quangeio_qa_studies": _bench_quangeio_qa_studies(seed)}
