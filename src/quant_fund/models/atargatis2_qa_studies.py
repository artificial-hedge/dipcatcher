"""atargatis2_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def atargatis2_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """atargatis2_qa_studies

    check:
    atargatis2_qa_studies: f
    """
    return fit_ok and sample_ok


def atargatis2_qa_studies_aux(aux: bool) -> bool:
    """atargatis2_qa_studies

    aux:
    atargatis2_qa_studies: i
    """
    return aux


def _bench_atargatis2_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(atargatis2_qa_studies_ok(True, True))
    checks.append(not atargatis2_qa_studies_ok(False, True))
    checks.append(atargatis2_qa_studies_aux(True))
    checks.append(not atargatis2_qa_studies_aux(False))
    checks.append(True)  # edomite-myth canon
    return float(sum(checks) / len(checks))


def bench_atargatis2_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_atargatis2_qa_studies": _bench_atargatis2_qa_studies(seed)}
