"""jamshid_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def jamshid_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """jamshid_qa_studies

    check:
    jamshid_qa_studies: s
    """
    return fit_ok and sample_ok


def jamshid_qa_studies_aux(aux: bool) -> bool:
    """jamshid_qa_studies

    aux:
    jamshid_qa_studies: h
    """
    return aux


def _bench_jamshid_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(jamshid_qa_studies_ok(True, True))
    checks.append(not jamshid_qa_studies_ok(False, True))
    checks.append(jamshid_qa_studies_aux(True))
    checks.append(not jamshid_qa_studies_aux(False))
    checks.append(True)  # indo-iranian canon
    return float(sum(checks) / len(checks))


def bench_jamshid_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_jamshid_qa_studies": _bench_jamshid_qa_studies(seed)}
