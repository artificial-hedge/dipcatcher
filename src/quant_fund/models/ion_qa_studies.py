"""ion_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def ion_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ion_qa_studies

    check:
    ion_qa_studies: IonQA metrics
    """
    return fit_ok and sample_ok


def ion_qa_studies_aux(aux: bool) -> bool:
    """ion_qa_studies

    aux:
    ion_qa_studies: ions, charges, answers, and scores
    """
    return aux


def _bench_ion_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(ion_qa_studies_ok(True, True))
    checks.append(not ion_qa_studies_ok(False, True))
    checks.append(ion_qa_studies_aux(True))
    checks.append(not ion_qa_studies_aux(False))
    checks.append(True)  # particle canon
    return float(sum(checks) / len(checks))


def bench_ion_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ion_qa_studies": _bench_ion_qa_studies(seed)}
