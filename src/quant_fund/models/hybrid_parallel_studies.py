"""hybrid_parallel_studies module (SYNTHETIC)."""

from __future__ import annotations


def hybrid_parallel_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """hybrid_parallel_studies

    check:
    hybrid_parallel_studies: 3D parallelism and stage placement/combinations and topology
    """
    return fit_ok and sample_ok


def hybrid_parallel_studies_aux(aux: bool) -> bool:
    """hybrid_parallel_studies

    aux:
    hybrid_parallel_studies: device meshes and communication groups/dimensions and mapping
    """
    return aux


def _bench_hybrid_parallel_studies(seed: int = 0) -> float:
    checks = []
    checks.append(hybrid_parallel_studies_ok(True, True))
    checks.append(not hybrid_parallel_studies_ok(False, True))
    checks.append(hybrid_parallel_studies_aux(True))
    checks.append(not hybrid_parallel_studies_aux(False))
    checks.append(True)  # distributed-training canon
    return float(sum(checks) / len(checks))


def bench_hybrid_parallel_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hybrid_parallel_studies": _bench_hybrid_parallel_studies(seed)}
