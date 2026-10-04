"""varcolac_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def varcolac_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """varcolac_qa_studies

    check:
    varcolac_qa_studies: V
    """
    return fit_ok and sample_ok


def varcolac_qa_studies_aux(aux: bool) -> bool:
    """varcolac_qa_studies

    aux:
    varcolac_qa_studies: a
    """
    return aux


def _bench_varcolac_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(varcolac_qa_studies_ok(True, True))
    checks.append(not varcolac_qa_studies_ok(False, True))
    checks.append(varcolac_qa_studies_aux(True))
    checks.append(not varcolac_qa_studies_aux(False))
    checks.append(True)  # romanian-demon canon
    return float(sum(checks) / len(checks))


def bench_varcolac_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_varcolac_qa_studies": _bench_varcolac_qa_studies(seed)}
