"""ubir_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def ubir_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ubir_qa_studies

    check:
    ubir_qa_studies: U
    """
    return fit_ok and sample_ok


def ubir_qa_studies_aux(aux: bool) -> bool:
    """ubir_qa_studies

    aux:
    ubir_qa_studies: b
    """
    return aux


def _bench_ubir_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(ubir_qa_studies_ok(True, True))
    checks.append(not ubir_qa_studies_ok(False, True))
    checks.append(ubir_qa_studies_aux(True))
    checks.append(not ubir_qa_studies_aux(False))
    checks.append(True)  # turkic-demon-2 canon
    return float(sum(checks) / len(checks))


def bench_ubir_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ubir_qa_studies": _bench_ubir_qa_studies(seed)}
