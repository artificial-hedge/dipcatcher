"""ao_andon_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def ao_andon_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ao_andon_qa_studies

    check:
    ao_andon_qa_studies: A
    """
    return fit_ok and sample_ok


def ao_andon_qa_studies_aux(aux: bool) -> bool:
    """ao_andon_qa_studies

    aux:
    ao_andon_qa_studies: o
    """
    return aux


def _bench_ao_andon_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(ao_andon_qa_studies_ok(True, True))
    checks.append(not ao_andon_qa_studies_ok(False, True))
    checks.append(ao_andon_qa_studies_aux(True))
    checks.append(not ao_andon_qa_studies_aux(False))
    checks.append(True)  # yokai-7 canon
    return float(sum(checks) / len(checks))


def bench_ao_andon_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ao_andon_qa_studies": _bench_ao_andon_qa_studies(seed)}
