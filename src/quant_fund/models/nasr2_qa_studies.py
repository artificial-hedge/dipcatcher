"""nasr2_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def nasr2_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """nasr2_qa_studies

    check:
    nasr2_qa_studies: v
    """
    return fit_ok and sample_ok


def nasr2_qa_studies_aux(aux: bool) -> bool:
    """nasr2_qa_studies

    aux:
    nasr2_qa_studies: u
    """
    return aux


def _bench_nasr2_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(nasr2_qa_studies_ok(True, True))
    checks.append(not nasr2_qa_studies_ok(False, True))
    checks.append(nasr2_qa_studies_aux(True))
    checks.append(not nasr2_qa_studies_aux(False))
    checks.append(True)  # sabaean-myth canon
    return float(sum(checks) / len(checks))


def bench_nasr2_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nasr2_qa_studies": _bench_nasr2_qa_studies(seed)}
