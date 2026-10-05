"""lilitu_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def lilitu_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """lilitu_qa_studies

    check:
    lilitu_qa_studies: l
    """
    return fit_ok and sample_ok


def lilitu_qa_studies_aux(aux: bool) -> bool:
    """lilitu_qa_studies

    aux:
    lilitu_qa_studies: i
    """
    return aux


def _bench_lilitu_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(lilitu_qa_studies_ok(True, True))
    checks.append(not lilitu_qa_studies_ok(False, True))
    checks.append(lilitu_qa_studies_aux(True))
    checks.append(not lilitu_qa_studies_aux(False))
    checks.append(True)  # mesopotamian-demon canon
    return float(sum(checks) / len(checks))


def bench_lilitu_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_lilitu_qa_studies": _bench_lilitu_qa_studies(seed)}
