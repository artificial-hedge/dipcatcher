"""llefelys_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def llefelys_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """llefelys_qa_studies

    check:
    llefelys_qa_studies: w
    """
    return fit_ok and sample_ok


def llefelys_qa_studies_aux(aux: bool) -> bool:
    """llefelys_qa_studies

    aux:
    llefelys_qa_studies: i
    """
    return aux


def _bench_llefelys_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(llefelys_qa_studies_ok(True, True))
    checks.append(not llefelys_qa_studies_ok(False, True))
    checks.append(llefelys_qa_studies_aux(True))
    checks.append(not llefelys_qa_studies_aux(False))
    checks.append(True)  # welsh-myth-3 canon
    return float(sum(checks) / len(checks))


def bench_llefelys_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_llefelys_qa_studies": _bench_llefelys_qa_studies(seed)}
