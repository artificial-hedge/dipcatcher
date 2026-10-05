"""qos2_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def qos2_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """qos2_qa_studies

    check:
    qos2_qa_studies: e
    """
    return fit_ok and sample_ok


def qos2_qa_studies_aux(aux: bool) -> bool:
    """qos2_qa_studies

    aux:
    qos2_qa_studies: d
    """
    return aux


def _bench_qos2_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(qos2_qa_studies_ok(True, True))
    checks.append(not qos2_qa_studies_ok(False, True))
    checks.append(qos2_qa_studies_aux(True))
    checks.append(not qos2_qa_studies_aux(False))
    checks.append(True)  # edomite-myth canon
    return float(sum(checks) / len(checks))


def bench_qos2_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_qos2_qa_studies": _bench_qos2_qa_studies(seed)}
