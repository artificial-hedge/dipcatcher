"""step_back_studies module (SYNTHETIC)."""

from __future__ import annotations


def step_back_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """step_back_studies

    check:
    step_back_studies: abstraction-first prompting/principles and instantiations
    """
    return fit_ok and sample_ok


def step_back_studies_aux(aux: bool) -> bool:
    """step_back_studies

    aux:
    step_back_studies: concept retrieval before answering/levels and grounding
    """
    return aux


def _bench_step_back_studies(seed: int = 0) -> float:
    checks = []
    checks.append(step_back_studies_ok(True, True))
    checks.append(not step_back_studies_ok(False, True))
    checks.append(step_back_studies_aux(True))
    checks.append(not step_back_studies_aux(False))
    checks.append(True)  # reasoning-prompt canon
    return float(sum(checks) / len(checks))


def bench_step_back_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_step_back_studies": _bench_step_back_studies(seed)}
