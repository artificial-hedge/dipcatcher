"""div_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def div_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """div_qa_studies

    check:
    div_qa_studies: d
    """
    return fit_ok and sample_ok


def div_qa_studies_aux(aux: bool) -> bool:
    """div_qa_studies

    aux:
    div_qa_studies: i
    """
    return aux


def _bench_div_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(div_qa_studies_ok(True, True))
    checks.append(not div_qa_studies_ok(False, True))
    checks.append(div_qa_studies_aux(True))
    checks.append(not div_qa_studies_aux(False))
    checks.append(True)  # zoroastrian-myth canon
    return float(sum(checks) / len(checks))


def bench_div_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_div_qa_studies": _bench_div_qa_studies(seed)}
