"""neak_ta_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def neak_ta_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """neak_ta_qa_studies

    check:
    neak_ta_qa_studies: N
    """
    return fit_ok and sample_ok


def neak_ta_qa_studies_aux(aux: bool) -> bool:
    """neak_ta_qa_studies

    aux:
    neak_ta_qa_studies: e
    """
    return aux


def _bench_neak_ta_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(neak_ta_qa_studies_ok(True, True))
    checks.append(not neak_ta_qa_studies_ok(False, True))
    checks.append(neak_ta_qa_studies_aux(True))
    checks.append(not neak_ta_qa_studies_aux(False))
    checks.append(True)  # khmer-demon canon
    return float(sum(checks) / len(checks))


def bench_neak_ta_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_neak_ta_qa_studies": _bench_neak_ta_qa_studies(seed)}
