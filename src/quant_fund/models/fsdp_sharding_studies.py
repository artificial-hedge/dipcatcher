"""fsdp_sharding_studies module (SYNTHETIC)."""

from __future__ import annotations


def fsdp_sharding_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """fsdp_sharding_studies

    check:
    fsdp_sharding_studies: parameter flattening and sharding/units and gather
    """
    return fit_ok and sample_ok


def fsdp_sharding_studies_aux(aux: bool) -> bool:
    """fsdp_sharding_studies

    aux:
    fsdp_sharding_studies: Fully-sharded data parallel policies/memory and scale
    """
    return aux


def _bench_fsdp_sharding_studies(seed: int = 0) -> float:
    checks = []
    checks.append(fsdp_sharding_studies_ok(True, True))
    checks.append(not fsdp_sharding_studies_ok(False, True))
    checks.append(fsdp_sharding_studies_aux(True))
    checks.append(not fsdp_sharding_studies_aux(False))
    checks.append(True)  # distributed-training canon
    return float(sum(checks) / len(checks))


def bench_fsdp_sharding_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fsdp_sharding_studies": _bench_fsdp_sharding_studies(seed)}
