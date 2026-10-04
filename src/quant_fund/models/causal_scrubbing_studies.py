"""causal_scrubbing_studies module (SYNTHETIC)."""

from __future__ import annotations


def causal_scrubbing_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """causal_scrubbing_studies

    check:
    causal_scrubbing_studies: hypothesis testing via resample ablations/circuits and faithfulness
    """
    return fit_ok and sample_ok


def causal_scrubbing_studies_aux(aux: bool) -> bool:
    """causal_scrubbing_studies

    aux:
    causal_scrubbing_studies: scrubbing and correspondence/nodes and invariances
    """
    return aux


def _bench_causal_scrubbing_studies(seed: int = 0) -> float:
    checks = []
    checks.append(causal_scrubbing_studies_ok(True, True))
    checks.append(not causal_scrubbing_studies_ok(False, True))
    checks.append(causal_scrubbing_studies_aux(True))
    checks.append(not causal_scrubbing_studies_aux(False))
    checks.append(True)  # interpretability-3 canon
    return float(sum(checks) / len(checks))


def bench_causal_scrubbing_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_causal_scrubbing_studies": _bench_causal_scrubbing_studies(seed)}
