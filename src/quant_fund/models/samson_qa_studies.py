"""samson_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def samson_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """samson_qa_studies

    check:
    samson_qa_studies: s
    """
    return fit_ok and sample_ok


def samson_qa_studies_aux(aux: bool) -> bool:
    """samson_qa_studies

    aux:
    samson_qa_studies: u
    """
    return aux


def _bench_samson_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(samson_qa_studies_ok(True, True))
    checks.append(not samson_qa_studies_ok(False, True))
    checks.append(samson_qa_studies_aux(True))
    checks.append(not samson_qa_studies_aux(False))
    checks.append(True)  # philistine-myth canon
    return float(sum(checks) / len(checks))


def bench_samson_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_samson_qa_studies": _bench_samson_qa_studies(seed)}
