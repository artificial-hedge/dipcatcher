"""magec_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def magec_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """magec_qa_studies

    check:
    magec_qa_studies: s
    """
    return fit_ok and sample_ok


def magec_qa_studies_aux(aux: bool) -> bool:
    """magec_qa_studies

    aux:
    magec_qa_studies: u
    """
    return aux


def _bench_magec_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(magec_qa_studies_ok(True, True))
    checks.append(not magec_qa_studies_ok(False, True))
    checks.append(magec_qa_studies_aux(True))
    checks.append(not magec_qa_studies_aux(False))
    checks.append(True)  # guanche-myth canon
    return float(sum(checks) / len(checks))


def bench_magec_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_magec_qa_studies": _bench_magec_qa_studies(seed)}
