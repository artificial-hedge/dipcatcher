"""obayifo_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def obayifo_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """obayifo_qa_studies

    check:
    obayifo_qa_studies: O
    """
    return fit_ok and sample_ok


def obayifo_qa_studies_aux(aux: bool) -> bool:
    """obayifo_qa_studies

    aux:
    obayifo_qa_studies: b
    """
    return aux


def _bench_obayifo_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(obayifo_qa_studies_ok(True, True))
    checks.append(not obayifo_qa_studies_ok(False, True))
    checks.append(obayifo_qa_studies_aux(True))
    checks.append(not obayifo_qa_studies_aux(False))
    checks.append(True)  # african-demon canon
    return float(sum(checks) / len(checks))


def bench_obayifo_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_obayifo_qa_studies": _bench_obayifo_qa_studies(seed)}
