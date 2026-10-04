"""deception_eval_studies module (SYNTHETIC)."""

from __future__ import annotations


def deception_eval_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """deception_eval_studies

    check:
    deception_eval_studies: alignment-faking and sandbagging probes/prompts and behaviors
    """
    return fit_ok and sample_ok


def deception_eval_studies_aux(aux: bool) -> bool:
    """deception_eval_studies

    aux:
    deception_eval_studies: honesty probes and sleeper-trigger evals/triggers and responses
    """
    return aux


def _bench_deception_eval_studies(seed: int = 0) -> float:
    checks = []
    checks.append(deception_eval_studies_ok(True, True))
    checks.append(not deception_eval_studies_ok(False, True))
    checks.append(deception_eval_studies_aux(True))
    checks.append(not deception_eval_studies_aux(False))
    checks.append(True)  # agent-safety canon
    return float(sum(checks) / len(checks))


def bench_deception_eval_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_deception_eval_studies": _bench_deception_eval_studies(seed)}
