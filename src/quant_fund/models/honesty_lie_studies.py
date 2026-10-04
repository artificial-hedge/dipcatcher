"""honesty_lie_studies module (SYNTHETIC)."""

from __future__ import annotations


def honesty_lie_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """honesty_lie_studies

    check:
    honesty_lie_studies: honesty/sycophancy metrics
    """
    return fit_ok and sample_ok


def honesty_lie_studies_aux(aux: bool) -> bool:
    """honesty_lie_studies

    aux:
    honesty_lie_studies: prompts, claims, labels, and accuracies
    """
    return aux


def _bench_honesty_lie_studies(seed: int = 0) -> float:
    checks = []
    checks.append(honesty_lie_studies_ok(True, True))
    checks.append(not honesty_lie_studies_ok(False, True))
    checks.append(honesty_lie_studies_aux(True))
    checks.append(not honesty_lie_studies_aux(False))
    checks.append(True)  # social-bias-eval canon
    return float(sum(checks) / len(checks))


def bench_honesty_lie_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_honesty_lie_studies": _bench_honesty_lie_studies(seed)}
