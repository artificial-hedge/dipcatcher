"""nakki_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def nakki_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """nakki_qa_studies

    check:
    nakki_qa_studies: N
    """
    return fit_ok and sample_ok


def nakki_qa_studies_aux(aux: bool) -> bool:
    """nakki_qa_studies

    aux:
    nakki_qa_studies: a
    """
    return aux


def _bench_nakki_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(nakki_qa_studies_ok(True, True))
    checks.append(not nakki_qa_studies_ok(False, True))
    checks.append(nakki_qa_studies_aux(True))
    checks.append(not nakki_qa_studies_aux(False))
    checks.append(True)  # finnish-demon canon
    return float(sum(checks) / len(checks))


def bench_nakki_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nakki_qa_studies": _bench_nakki_qa_studies(seed)}
