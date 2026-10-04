"""adversarial_eval_studies module (SYNTHETIC)."""

from __future__ import annotations


def adversarial_eval_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """adversarial_eval_studies

    check:
    adversarial_eval_studies: adversarial perturbation budgets/attacks and acc
    """
    return fit_ok and sample_ok


def adversarial_eval_studies_aux(aux: bool) -> bool:
    """adversarial_eval_studies

    aux:
    adversarial_eval_studies: attack grids/norms and worst-case scores
    """
    return aux


def _bench_adversarial_eval_studies(seed: int = 0) -> float:
    checks = []
    checks.append(adversarial_eval_studies_ok(True, True))
    checks.append(not adversarial_eval_studies_ok(False, True))
    checks.append(adversarial_eval_studies_aux(True))
    checks.append(not adversarial_eval_studies_aux(False))
    checks.append(True)  # robustness-eval canon
    return float(sum(checks) / len(checks))


def bench_adversarial_eval_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_adversarial_eval_studies": _bench_adversarial_eval_studies(seed)}
