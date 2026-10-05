"""barbatos_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def barbatos_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """barbatos_qa_studies

    check:
    barbatos_qa_studies: B
    """
    return fit_ok and sample_ok


def barbatos_qa_studies_aux(aux: bool) -> bool:
    """barbatos_qa_studies

    aux:
    barbatos_qa_studies: a
    """
    return aux


def _bench_barbatos_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(barbatos_qa_studies_ok(True, True))
    checks.append(not barbatos_qa_studies_ok(False, True))
    checks.append(barbatos_qa_studies_aux(True))
    checks.append(not barbatos_qa_studies_aux(False))
    checks.append(True)  # goetic-legion canon
    return float(sum(checks) / len(checks))


def bench_barbatos_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_barbatos_qa_studies": _bench_barbatos_qa_studies(seed)}
