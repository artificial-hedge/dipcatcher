"""kilmoulis_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def kilmoulis_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """kilmoulis_qa_studies

    check:
    kilmoulis_qa_studies: K
    """
    return fit_ok and sample_ok


def kilmoulis_qa_studies_aux(aux: bool) -> bool:
    """kilmoulis_qa_studies

    aux:
    kilmoulis_qa_studies: i
    """
    return aux


def _bench_kilmoulis_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(kilmoulis_qa_studies_ok(True, True))
    checks.append(not kilmoulis_qa_studies_ok(False, True))
    checks.append(kilmoulis_qa_studies_aux(True))
    checks.append(not kilmoulis_qa_studies_aux(False))
    checks.append(True)  # celtic-demon-3 canon
    return float(sum(checks) / len(checks))


def bench_kilmoulis_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kilmoulis_qa_studies": _bench_kilmoulis_qa_studies(seed)}
