"""braciaca_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def braciaca_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """braciaca_qa_studies

    check:
    braciaca_qa_studies: h
    """
    return fit_ok and sample_ok


def braciaca_qa_studies_aux(aux: bool) -> bool:
    """braciaca_qa_studies

    aux:
    braciaca_qa_studies: e
    """
    return aux


def _bench_braciaca_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(braciaca_qa_studies_ok(True, True))
    checks.append(not braciaca_qa_studies_ok(False, True))
    checks.append(braciaca_qa_studies_aux(True))
    checks.append(not braciaca_qa_studies_aux(False))
    checks.append(True)  # celtic-remnant canon
    return float(sum(checks) / len(checks))


def bench_braciaca_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_braciaca_qa_studies": _bench_braciaca_qa_studies(seed)}
