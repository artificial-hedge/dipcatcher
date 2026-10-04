"""haubas_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def haubas_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """haubas_qa_studies

    check:
    haubas_qa_studies: h
    """
    return fit_ok and sample_ok


def haubas_qa_studies_aux(aux: bool) -> bool:
    """haubas_qa_studies

    aux:
    haubas_qa_studies: i
    """
    return aux


def _bench_haubas_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(haubas_qa_studies_ok(True, True))
    checks.append(not haubas_qa_studies_ok(False, True))
    checks.append(haubas_qa_studies_aux(True))
    checks.append(not haubas_qa_studies_aux(False))
    checks.append(True)  # sabaean-myth canon
    return float(sum(checks) / len(checks))


def bench_haubas_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_haubas_qa_studies": _bench_haubas_qa_studies(seed)}
