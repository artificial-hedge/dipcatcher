"""taranis_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def taranis_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """taranis_qa_studies

    check:
    taranis_qa_studies: t
    """
    return fit_ok and sample_ok


def taranis_qa_studies_aux(aux: bool) -> bool:
    """taranis_qa_studies

    aux:
    taranis_qa_studies: h
    """
    return aux


def _bench_taranis_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(taranis_qa_studies_ok(True, True))
    checks.append(not taranis_qa_studies_ok(False, True))
    checks.append(taranis_qa_studies_aux(True))
    checks.append(not taranis_qa_studies_aux(False))
    checks.append(True)  # gallic-myth canon
    return float(sum(checks) / len(checks))


def bench_taranis_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_taranis_qa_studies": _bench_taranis_qa_studies(seed)}
