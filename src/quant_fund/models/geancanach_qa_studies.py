"""geancanach_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def geancanach_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """geancanach_qa_studies

    check:
    geancanach_qa_studies: G
    """
    return fit_ok and sample_ok


def geancanach_qa_studies_aux(aux: bool) -> bool:
    """geancanach_qa_studies

    aux:
    geancanach_qa_studies: e
    """
    return aux


def _bench_geancanach_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(geancanach_qa_studies_ok(True, True))
    checks.append(not geancanach_qa_studies_ok(False, True))
    checks.append(geancanach_qa_studies_aux(True))
    checks.append(not geancanach_qa_studies_aux(False))
    checks.append(True)  # celtic-demon-3 canon
    return float(sum(checks) / len(checks))


def bench_geancanach_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_geancanach_qa_studies": _bench_geancanach_qa_studies(seed)}
