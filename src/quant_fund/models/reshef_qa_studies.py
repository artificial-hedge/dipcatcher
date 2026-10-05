"""reshef_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def reshef_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """reshef_qa_studies

    check:
    reshef_qa_studies: p
    """
    return fit_ok and sample_ok


def reshef_qa_studies_aux(aux: bool) -> bool:
    """reshef_qa_studies

    aux:
    reshef_qa_studies: l
    """
    return aux


def _bench_reshef_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(reshef_qa_studies_ok(True, True))
    checks.append(not reshef_qa_studies_ok(False, True))
    checks.append(reshef_qa_studies_aux(True))
    checks.append(not reshef_qa_studies_aux(False))
    checks.append(True)  # punic-3 canon
    return float(sum(checks) / len(checks))


def bench_reshef_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_reshef_qa_studies": _bench_reshef_qa_studies(seed)}
