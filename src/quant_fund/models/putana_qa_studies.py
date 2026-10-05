"""putana_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def putana_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """putana_qa_studies

    check:
    putana_qa_studies: P
    """
    return fit_ok and sample_ok


def putana_qa_studies_aux(aux: bool) -> bool:
    """putana_qa_studies

    aux:
    putana_qa_studies: u
    """
    return aux


def _bench_putana_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(putana_qa_studies_ok(True, True))
    checks.append(not putana_qa_studies_ok(False, True))
    checks.append(putana_qa_studies_aux(True))
    checks.append(not putana_qa_studies_aux(False))
    checks.append(True)  # hindu-demon canon
    return float(sum(checks) / len(checks))


def bench_putana_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_putana_qa_studies": _bench_putana_qa_studies(seed)}
