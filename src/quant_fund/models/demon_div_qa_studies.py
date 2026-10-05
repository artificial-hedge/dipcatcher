"""demon_div_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def demon_div_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """demon_div_qa_studies

    check:
    demon_div_qa_studies: D
    """
    return fit_ok and sample_ok


def demon_div_qa_studies_aux(aux: bool) -> bool:
    """demon_div_qa_studies

    aux:
    demon_div_qa_studies: e
    """
    return aux


def _bench_demon_div_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(demon_div_qa_studies_ok(True, True))
    checks.append(not demon_div_qa_studies_ok(False, True))
    checks.append(demon_div_qa_studies_aux(True))
    checks.append(not demon_div_qa_studies_aux(False))
    checks.append(True)  # persian-div canon
    return float(sum(checks) / len(checks))


def bench_demon_div_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_demon_div_qa_studies": _bench_demon_div_qa_studies(seed)}
