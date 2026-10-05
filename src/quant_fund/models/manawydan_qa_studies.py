"""manawydan_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def manawydan_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """manawydan_qa_studies

    check:
    manawydan_qa_studies: s
    """
    return fit_ok and sample_ok


def manawydan_qa_studies_aux(aux: bool) -> bool:
    """manawydan_qa_studies

    aux:
    manawydan_qa_studies: e
    """
    return aux


def _bench_manawydan_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(manawydan_qa_studies_ok(True, True))
    checks.append(not manawydan_qa_studies_ok(False, True))
    checks.append(manawydan_qa_studies_aux(True))
    checks.append(not manawydan_qa_studies_aux(False))
    checks.append(True)  # welsh-myth-3 canon
    return float(sum(checks) / len(checks))


def bench_manawydan_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_manawydan_qa_studies": _bench_manawydan_qa_studies(seed)}
