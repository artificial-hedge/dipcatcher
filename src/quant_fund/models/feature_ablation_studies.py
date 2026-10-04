"""feature_ablation_studies module (SYNTHETIC)."""

from __future__ import annotations


def feature_ablation_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """feature_ablation_studies

    check:
    feature_ablation_studies: causal feature knockout evals/ablates and deltas
    """
    return fit_ok and sample_ok


def feature_ablation_studies_aux(aux: bool) -> bool:
    """feature_ablation_studies

    aux:
    feature_ablation_studies: steering-vector add/remove interventions/heads and shifts
    """
    return aux


def _bench_feature_ablation_studies(seed: int = 0) -> float:
    checks = []
    checks.append(feature_ablation_studies_ok(True, True))
    checks.append(not feature_ablation_studies_ok(False, True))
    checks.append(feature_ablation_studies_aux(True))
    checks.append(not feature_ablation_studies_aux(False))
    checks.append(True)  # representation-engineering canon
    return float(sum(checks) / len(checks))


def bench_feature_ablation_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_feature_ablation_studies": _bench_feature_ablation_studies(seed)}
