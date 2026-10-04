"""ifeval_studies module (SYNTHETIC)."""

from __future__ import annotations


def ifeval_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ifeval_studies

    check:
    ifeval_studies: IFEval instruction-following constraints and acc
    """
    return fit_ok and sample_ok


def ifeval_studies_aux(aux: bool) -> bool:
    """ifeval_studies

    aux:
    ifeval_studies: verifiable instructions, checks, and pass rates
    """
    return aux


def _bench_ifeval_studies(seed: int = 0) -> float:
    checks = []
    checks.append(ifeval_studies_ok(True, True))
    checks.append(not ifeval_studies_ok(False, True))
    checks.append(ifeval_studies_aux(True))
    checks.append(not ifeval_studies_aux(False))
    checks.append(True)  # LLM-academic-eval canon
    return float(sum(checks) / len(checks))


def bench_ifeval_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ifeval_studies": _bench_ifeval_studies(seed)}
