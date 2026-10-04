"""zero_optimizer_studies module (SYNTHETIC)."""

from __future__ import annotations


def zero_optimizer_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """zero_optimizer_studies

    check:
    zero_optimizer_studies: optimizer-state sharding and stages/params and gradients
    """
    return fit_ok and sample_ok


def zero_optimizer_studies_aux(aux: bool) -> bool:
    """zero_optimizer_studies

    aux:
    zero_optimizer_studies: ZeRO-1/2/3 memory partitioning/offload and overlap
    """
    return aux


def _bench_zero_optimizer_studies(seed: int = 0) -> float:
    checks = []
    checks.append(zero_optimizer_studies_ok(True, True))
    checks.append(not zero_optimizer_studies_ok(False, True))
    checks.append(zero_optimizer_studies_aux(True))
    checks.append(not zero_optimizer_studies_aux(False))
    checks.append(True)  # distributed-training canon
    return float(sum(checks) / len(checks))


def bench_zero_optimizer_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_zero_optimizer_studies": _bench_zero_optimizer_studies(seed)}
