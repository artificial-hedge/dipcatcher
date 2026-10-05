"""buggane_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def buggane_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """buggane_qa_studies

    check:
    buggane_qa_studies: o
    """
    return fit_ok and sample_ok


def buggane_qa_studies_aux(aux: bool) -> bool:
    """buggane_qa_studies

    aux:
    buggane_qa_studies: g
    """
    return aux


def _bench_buggane_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(buggane_qa_studies_ok(True, True))
    checks.append(not buggane_qa_studies_ok(False, True))
    checks.append(buggane_qa_studies_aux(True))
    checks.append(not buggane_qa_studies_aux(False))
    checks.append(True)  # manx-myth canon
    return float(sum(checks) / len(checks))


def bench_buggane_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_buggane_qa_studies": _bench_buggane_qa_studies(seed)}
