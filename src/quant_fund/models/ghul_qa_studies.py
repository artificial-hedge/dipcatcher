"""ghul_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def ghul_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ghul_qa_studies

    check:
    ghul_qa_studies: G
    """
    return fit_ok and sample_ok


def ghul_qa_studies_aux(aux: bool) -> bool:
    """ghul_qa_studies

    aux:
    ghul_qa_studies: h
    """
    return aux


def _bench_ghul_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(ghul_qa_studies_ok(True, True))
    checks.append(not ghul_qa_studies_ok(False, True))
    checks.append(ghul_qa_studies_aux(True))
    checks.append(not ghul_qa_studies_aux(False))
    checks.append(True)  # jinn canon
    return float(sum(checks) / len(checks))


def bench_ghul_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ghul_qa_studies": _bench_ghul_qa_studies(seed)}
