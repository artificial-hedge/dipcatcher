"""dialect_bias_studies module (SYNTHETIC)."""

from __future__ import annotations


def dialect_bias_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """dialect_bias_studies

    check:
    dialect_bias_studies: Dialect-bias probe metrics
    """
    return fit_ok and sample_ok


def dialect_bias_studies_aux(aux: bool) -> bool:
    """dialect_bias_studies

    aux:
    dialect_bias_studies: utterances, dialects, labels, and accuracies
    """
    return aux


def _bench_dialect_bias_studies(seed: int = 0) -> float:
    checks = []
    checks.append(dialect_bias_studies_ok(True, True))
    checks.append(not dialect_bias_studies_ok(False, True))
    checks.append(dialect_bias_studies_aux(True))
    checks.append(not dialect_bias_studies_aux(False))
    checks.append(True)  # rumor-bias canon
    return float(sum(checks) / len(checks))


def bench_dialect_bias_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dialect_bias_studies": _bench_dialect_bias_studies(seed)}
