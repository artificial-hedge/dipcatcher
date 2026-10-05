"""sunakake_babaa_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def sunakake_babaa_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """sunakake_babaa_qa_studies

    check:
    sunakake_babaa_qa_studies: S
    """
    return fit_ok and sample_ok


def sunakake_babaa_qa_studies_aux(aux: bool) -> bool:
    """sunakake_babaa_qa_studies

    aux:
    sunakake_babaa_qa_studies: u
    """
    return aux


def _bench_sunakake_babaa_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(sunakake_babaa_qa_studies_ok(True, True))
    checks.append(not sunakake_babaa_qa_studies_ok(False, True))
    checks.append(sunakake_babaa_qa_studies_aux(True))
    checks.append(not sunakake_babaa_qa_studies_aux(False))
    checks.append(True)  # yokai-8 canon
    return float(sum(checks) / len(checks))


def bench_sunakake_babaa_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sunakake_babaa_qa_studies": _bench_sunakake_babaa_qa_studies(seed)}
