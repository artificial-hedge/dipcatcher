"""weight_share_studies module (SYNTHETIC)."""

from __future__ import annotations


def weight_share_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """weight_share_studies

    check:
    weight_share_studies: weight clustering and shared-centroid nets/clusters and lookups
    """
    return fit_ok and sample_ok


def weight_share_studies_aux(aux: bool) -> bool:
    """weight_share_studies

    aux:
    weight_share_studies: hashed/centroid parameter sharing/buckets and collisions
    """
    return aux


def _bench_weight_share_studies(seed: int = 0) -> float:
    checks = []
    checks.append(weight_share_studies_ok(True, True))
    checks.append(not weight_share_studies_ok(False, True))
    checks.append(weight_share_studies_aux(True))
    checks.append(not weight_share_studies_aux(False))
    checks.append(True)  # quantization/compression canon
    return float(sum(checks) / len(checks))


def bench_weight_share_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_weight_share_studies": _bench_weight_share_studies(seed)}
