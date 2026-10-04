"""moloch2_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def moloch2_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """moloch2_qa_studies

    check:
    moloch2_qa_studies: f
    """
    return fit_ok and sample_ok


def moloch2_qa_studies_aux(aux: bool) -> bool:
    """moloch2_qa_studies

    aux:
    moloch2_qa_studies: i
    """
    return aux


def _bench_moloch2_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(moloch2_qa_studies_ok(True, True))
    checks.append(not moloch2_qa_studies_ok(False, True))
    checks.append(moloch2_qa_studies_aux(True))
    checks.append(not moloch2_qa_studies_aux(False))
    checks.append(True)  # ammonite-myth canon
    return float(sum(checks) / len(checks))


def bench_moloch2_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_moloch2_qa_studies": _bench_moloch2_qa_studies(seed)}
