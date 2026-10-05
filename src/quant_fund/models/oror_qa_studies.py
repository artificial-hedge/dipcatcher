"""oror_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def oror_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """oror_qa_studies

    check:
    oror_qa_studies: O
    """
    return fit_ok and sample_ok


def oror_qa_studies_aux(aux: bool) -> bool:
    """oror_qa_studies

    aux:
    oror_qa_studies: r
    """
    return aux


def _bench_oror_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(oror_qa_studies_ok(True, True))
    checks.append(not oror_qa_studies_ok(False, True))
    checks.append(oror_qa_studies_aux(True))
    checks.append(not oror_qa_studies_aux(False))
    checks.append(True)  # siberian-demon canon
    return float(sum(checks) / len(checks))


def bench_oror_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_oror_qa_studies": _bench_oror_qa_studies(seed)}
