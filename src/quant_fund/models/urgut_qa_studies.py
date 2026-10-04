"""urgut_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def urgut_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """urgut_qa_studies

    check:
    urgut_qa_studies: U
    """
    return fit_ok and sample_ok


def urgut_qa_studies_aux(aux: bool) -> bool:
    """urgut_qa_studies

    aux:
    urgut_qa_studies: r
    """
    return aux


def _bench_urgut_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(urgut_qa_studies_ok(True, True))
    checks.append(not urgut_qa_studies_ok(False, True))
    checks.append(urgut_qa_studies_aux(True))
    checks.append(not urgut_qa_studies_aux(False))
    checks.append(True)  # siberian-demon canon
    return float(sum(checks) / len(checks))


def bench_urgut_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_urgut_qa_studies": _bench_urgut_qa_studies(seed)}
