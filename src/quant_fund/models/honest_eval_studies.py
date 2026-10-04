"""honest_eval_studies module (SYNTHETIC)."""

from __future__ import annotations


def honest_eval_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """honest_eval_studies

    check:
    honest_eval_studies: HONEST hurtful-completion metrics
    """
    return fit_ok and sample_ok


def honest_eval_studies_aux(aux: bool) -> bool:
    """honest_eval_studies

    aux:
    honest_eval_studies: prompts, completions, and hurtfulness scores
    """
    return aux


def _bench_honest_eval_studies(seed: int = 0) -> float:
    checks = []
    checks.append(honest_eval_studies_ok(True, True))
    checks.append(not honest_eval_studies_ok(False, True))
    checks.append(honest_eval_studies_aux(True))
    checks.append(not honest_eval_studies_aux(False))
    checks.append(True)  # safety-alignment-2 canon
    return float(sum(checks) / len(checks))


def bench_honest_eval_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_honest_eval_studies": _bench_honest_eval_studies(seed)}
