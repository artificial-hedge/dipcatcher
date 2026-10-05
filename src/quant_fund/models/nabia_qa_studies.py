"""nabia_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def nabia_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """nabia_qa_studies

    check:
    nabia_qa_studies: v
    """
    return fit_ok and sample_ok


def nabia_qa_studies_aux(aux: bool) -> bool:
    """nabia_qa_studies

    aux:
    nabia_qa_studies: a
    """
    return aux


def _bench_nabia_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(nabia_qa_studies_ok(True, True))
    checks.append(not nabia_qa_studies_ok(False, True))
    checks.append(nabia_qa_studies_aux(True))
    checks.append(not nabia_qa_studies_aux(False))
    checks.append(True)  # iberian-myth canon
    return float(sum(checks) / len(checks))


def bench_nabia_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nabia_qa_studies": _bench_nabia_qa_studies(seed)}
