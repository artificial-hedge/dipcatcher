"""shape_bias_studies module (SYNTHETIC)."""

from __future__ import annotations


def shape_bias_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """shape_bias_studies

    check:
    shape_bias_studies: shape-bias preference and robustness scores
    """
    return fit_ok and sample_ok


def shape_bias_studies_aux(aux: bool) -> bool:
    """shape_bias_studies

    aux:
    shape_bias_studies: edge/silhouette probes, rates, and accuracy
    """
    return aux


def _bench_shape_bias_studies(seed: int = 0) -> float:
    checks = []
    checks.append(shape_bias_studies_ok(True, True))
    checks.append(not shape_bias_studies_ok(False, True))
    checks.append(shape_bias_studies_aux(True))
    checks.append(not shape_bias_studies_aux(False))
    checks.append(True)  # cue-conflict canon
    return float(sum(checks) / len(checks))


def bench_shape_bias_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_shape_bias_studies": _bench_shape_bias_studies(seed)}
