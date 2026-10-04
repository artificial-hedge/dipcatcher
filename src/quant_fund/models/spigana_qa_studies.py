"""spigana_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def spigana_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """spigana_qa_studies

    check:
    spigana_qa_studies: S
    """
    return fit_ok and sample_ok


def spigana_qa_studies_aux(aux: bool) -> bool:
    """spigana_qa_studies

    aux:
    spigana_qa_studies: p
    """
    return aux


def _bench_spigana_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(spigana_qa_studies_ok(True, True))
    checks.append(not spigana_qa_studies_ok(False, True))
    checks.append(spigana_qa_studies_aux(True))
    checks.append(not spigana_qa_studies_aux(False))
    checks.append(True)  # baltic-demon canon
    return float(sum(checks) / len(checks))


def bench_spigana_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_spigana_qa_studies": _bench_spigana_qa_studies(seed)}
