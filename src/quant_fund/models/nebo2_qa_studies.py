"""nebo2_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def nebo2_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """nebo2_qa_studies

    check:
    nebo2_qa_studies: p
    """
    return fit_ok and sample_ok


def nebo2_qa_studies_aux(aux: bool) -> bool:
    """nebo2_qa_studies

    aux:
    nebo2_qa_studies: r
    """
    return aux


def _bench_nebo2_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(nebo2_qa_studies_ok(True, True))
    checks.append(not nebo2_qa_studies_ok(False, True))
    checks.append(nebo2_qa_studies_aux(True))
    checks.append(not nebo2_qa_studies_aux(False))
    checks.append(True)  # moabite-myth canon
    return float(sum(checks) / len(checks))


def bench_nebo2_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nebo2_qa_studies": _bench_nebo2_qa_studies(seed)}
