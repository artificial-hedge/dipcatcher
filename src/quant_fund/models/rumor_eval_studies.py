"""rumor_eval_studies module (SYNTHETIC)."""

from __future__ import annotations


def rumor_eval_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """rumor_eval_studies

    check:
    rumor_eval_studies: rumor-detection metrics
    """
    return fit_ok and sample_ok


def rumor_eval_studies_aux(aux: bool) -> bool:
    """rumor_eval_studies

    aux:
    rumor_eval_studies: posts, threads, labels, and accuracies
    """
    return aux


def _bench_rumor_eval_studies(seed: int = 0) -> float:
    checks = []
    checks.append(rumor_eval_studies_ok(True, True))
    checks.append(not rumor_eval_studies_ok(False, True))
    checks.append(rumor_eval_studies_aux(True))
    checks.append(not rumor_eval_studies_aux(False))
    checks.append(True)  # misinformation canon
    return float(sum(checks) / len(checks))


def bench_rumor_eval_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_rumor_eval_studies": _bench_rumor_eval_studies(seed)}
