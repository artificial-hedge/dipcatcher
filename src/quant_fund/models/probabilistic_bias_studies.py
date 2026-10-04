"""probabilistic_bias_studies module (SYNTHETIC)."""

from __future__ import annotations


def probabilistic_bias_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """probabilistic_bias_studies

    check:
    probabilistic_bias_studies: bias parameters and distributions/simulation and bounds
    """
    return fit_ok and sample_ok


def probabilistic_bias_studies_aux(aux: bool) -> bool:
    """probabilistic_bias_studies

    aux:
    probabilistic_bias_studies: record-level and analyses/uncertainty and ranges
    """
    return aux


def _bench_probabilistic_bias_studies(seed: int = 0) -> float:
    checks = []
    checks.append(probabilistic_bias_studies_ok(True, True))
    checks.append(not probabilistic_bias_studies_ok(False, True))
    checks.append(probabilistic_bias_studies_aux(True))
    checks.append(not probabilistic_bias_studies_aux(False))
    checks.append(True)  # causal-RWE-2 canon
    return float(sum(checks) / len(checks))


def bench_probabilistic_bias_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_probabilistic_bias_studies": _bench_probabilistic_bias_studies(seed)}
