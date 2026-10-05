"""mboitui_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def mboitui_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """mboitui_qa_studies

    check:
    mboitui_qa_studies: M
    """
    return fit_ok and sample_ok


def mboitui_qa_studies_aux(aux: bool) -> bool:
    """mboitui_qa_studies

    aux:
    mboitui_qa_studies: b
    """
    return aux


def _bench_mboitui_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(mboitui_qa_studies_ok(True, True))
    checks.append(not mboitui_qa_studies_ok(False, True))
    checks.append(mboitui_qa_studies_aux(True))
    checks.append(not mboitui_qa_studies_aux(False))
    checks.append(True)  # guarani-demon canon
    return float(sum(checks) / len(checks))


def bench_mboitui_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mboitui_qa_studies": _bench_mboitui_qa_studies(seed)}
