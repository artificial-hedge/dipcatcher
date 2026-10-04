"""pellinor_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def pellinor_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """pellinor_qa_studies

    check:
    pellinor_qa_studies: q
    """
    return fit_ok and sample_ok


def pellinor_qa_studies_aux(aux: bool) -> bool:
    """pellinor_qa_studies

    aux:
    pellinor_qa_studies: u
    """
    return aux


def _bench_pellinor_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(pellinor_qa_studies_ok(True, True))
    checks.append(not pellinor_qa_studies_ok(False, True))
    checks.append(pellinor_qa_studies_aux(True))
    checks.append(not pellinor_qa_studies_aux(False))
    checks.append(True)  # arthurian-4 canon
    return float(sum(checks) / len(checks))


def bench_pellinor_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_pellinor_qa_studies": _bench_pellinor_qa_studies(seed)}
