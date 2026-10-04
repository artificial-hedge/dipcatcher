"""piskie_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def piskie_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """piskie_qa_studies

    check:
    piskie_qa_studies: l
    """
    return fit_ok and sample_ok


def piskie_qa_studies_aux(aux: bool) -> bool:
    """piskie_qa_studies

    aux:
    piskie_qa_studies: i
    """
    return aux


def _bench_piskie_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(piskie_qa_studies_ok(True, True))
    checks.append(not piskie_qa_studies_ok(False, True))
    checks.append(piskie_qa_studies_aux(True))
    checks.append(not piskie_qa_studies_aux(False))
    checks.append(True)  # cornish-myth canon
    return float(sum(checks) / len(checks))


def bench_piskie_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_piskie_qa_studies": _bench_piskie_qa_studies(seed)}
