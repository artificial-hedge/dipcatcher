"""marchosias_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def marchosias_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """marchosias_qa_studies

    check:
    marchosias_qa_studies: M
    """
    return fit_ok and sample_ok


def marchosias_qa_studies_aux(aux: bool) -> bool:
    """marchosias_qa_studies

    aux:
    marchosias_qa_studies: a
    """
    return aux


def _bench_marchosias_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(marchosias_qa_studies_ok(True, True))
    checks.append(not marchosias_qa_studies_ok(False, True))
    checks.append(marchosias_qa_studies_aux(True))
    checks.append(not marchosias_qa_studies_aux(False))
    checks.append(True)  # goetic-throne canon
    return float(sum(checks) / len(checks))


def bench_marchosias_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_marchosias_qa_studies": _bench_marchosias_qa_studies(seed)}
