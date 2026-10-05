"""dunter_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def dunter_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """dunter_qa_studies

    check:
    dunter_qa_studies: D
    """
    return fit_ok and sample_ok


def dunter_qa_studies_aux(aux: bool) -> bool:
    """dunter_qa_studies

    aux:
    dunter_qa_studies: u
    """
    return aux


def _bench_dunter_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(dunter_qa_studies_ok(True, True))
    checks.append(not dunter_qa_studies_ok(False, True))
    checks.append(dunter_qa_studies_aux(True))
    checks.append(not dunter_qa_studies_aux(False))
    checks.append(True)  # celtic-demon-2 canon
    return float(sum(checks) / len(checks))


def bench_dunter_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dunter_qa_studies": _bench_dunter_qa_studies(seed)}
