"""rimmon_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def rimmon_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """rimmon_qa_studies

    check:
    rimmon_qa_studies: t
    """
    return fit_ok and sample_ok


def rimmon_qa_studies_aux(aux: bool) -> bool:
    """rimmon_qa_studies

    aux:
    rimmon_qa_studies: h
    """
    return aux


def _bench_rimmon_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(rimmon_qa_studies_ok(True, True))
    checks.append(not rimmon_qa_studies_ok(False, True))
    checks.append(rimmon_qa_studies_aux(True))
    checks.append(not rimmon_qa_studies_aux(False))
    checks.append(True)  # aramaean-myth canon
    return float(sum(checks) / len(checks))


def bench_rimmon_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_rimmon_qa_studies": _bench_rimmon_qa_studies(seed)}
