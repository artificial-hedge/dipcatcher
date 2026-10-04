"""deliberative_alignment_studies module (SYNTHETIC)."""

from __future__ import annotations


def deliberative_alignment_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """deliberative_alignment_studies

    check:
    deliberative_alignment_studies: reasoned safety spec and rule inference/principles and responses
    """
    return fit_ok and sample_ok


def deliberative_alignment_studies_aux(aux: bool) -> bool:
    """deliberative_alignment_studies

    aux:
    deliberative_alignment_studies: chain-of-thought policy checking/deliberation and refusal
    """
    return aux


def _bench_deliberative_alignment_studies(seed: int = 0) -> float:
    checks = []
    checks.append(deliberative_alignment_studies_ok(True, True))
    checks.append(not deliberative_alignment_studies_ok(False, True))
    checks.append(deliberative_alignment_studies_aux(True))
    checks.append(not deliberative_alignment_studies_aux(False))
    checks.append(True)  # scalable-oversight canon
    return float(sum(checks) / len(checks))


def bench_deliberative_alignment_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_deliberative_alignment_studies": _bench_deliberative_alignment_studies(seed)}
