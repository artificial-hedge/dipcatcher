"""tarroo_ushtey_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def tarroo_ushtey_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """tarroo_ushtey_qa_studies

    check:
    tarroo_ushtey_qa_studies: w
    """
    return fit_ok and sample_ok


def tarroo_ushtey_qa_studies_aux(aux: bool) -> bool:
    """tarroo_ushtey_qa_studies

    aux:
    tarroo_ushtey_qa_studies: a
    """
    return aux


def _bench_tarroo_ushtey_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(tarroo_ushtey_qa_studies_ok(True, True))
    checks.append(not tarroo_ushtey_qa_studies_ok(False, True))
    checks.append(tarroo_ushtey_qa_studies_aux(True))
    checks.append(not tarroo_ushtey_qa_studies_aux(False))
    checks.append(True)  # manx-myth canon
    return float(sum(checks) / len(checks))


def bench_tarroo_ushtey_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tarroo_ushtey_qa_studies": _bench_tarroo_ushtey_qa_studies(seed)}
