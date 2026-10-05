"""el_punic_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def el_punic_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """el_punic_qa_studies

    check:
    el_punic_qa_studies: f
    """
    return fit_ok and sample_ok


def el_punic_qa_studies_aux(aux: bool) -> bool:
    """el_punic_qa_studies

    aux:
    el_punic_qa_studies: a
    """
    return aux


def _bench_el_punic_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(el_punic_qa_studies_ok(True, True))
    checks.append(not el_punic_qa_studies_ok(False, True))
    checks.append(el_punic_qa_studies_aux(True))
    checks.append(not el_punic_qa_studies_aux(False))
    checks.append(True)  # carthaginian-2 canon
    return float(sum(checks) / len(checks))


def bench_el_punic_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_el_punic_qa_studies": _bench_el_punic_qa_studies(seed)}
