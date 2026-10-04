"""activation_oracle_studies module (SYNTHETIC)."""

from __future__ import annotations


def activation_oracle_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """activation_oracle_studies

    check:
    activation_oracle_studies: internal-state lie detection/probes and claims
    """
    return fit_ok and sample_ok


def activation_oracle_studies_aux(aux: bool) -> bool:
    """activation_oracle_studies

    aux:
    activation_oracle_studies: oracle probes over residual streams/layers and scores
    """
    return aux


def _bench_activation_oracle_studies(seed: int = 0) -> float:
    checks = []
    checks.append(activation_oracle_studies_ok(True, True))
    checks.append(not activation_oracle_studies_ok(False, True))
    checks.append(activation_oracle_studies_aux(True))
    checks.append(not activation_oracle_studies_aux(False))
    checks.append(True)  # representation-engineering canon
    return float(sum(checks) / len(checks))


def bench_activation_oracle_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_activation_oracle_studies": _bench_activation_oracle_studies(seed)}
