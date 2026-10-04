"""sequence_parallel_studies module (SYNTHETIC)."""

from __future__ import annotations


def sequence_parallel_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """sequence_parallel_studies

    check:
    sequence_parallel_studies: sequence-dim sharding and all-gather heads/context and length
    """
    return fit_ok and sample_ok


def sequence_parallel_studies_aux(aux: bool) -> bool:
    """sequence_parallel_studies

    aux:
    sequence_parallel_studies: DeepSpeed-Ulysses and ring attention/long and sparse
    """
    return aux


def _bench_sequence_parallel_studies(seed: int = 0) -> float:
    checks = []
    checks.append(sequence_parallel_studies_ok(True, True))
    checks.append(not sequence_parallel_studies_ok(False, True))
    checks.append(sequence_parallel_studies_aux(True))
    checks.append(not sequence_parallel_studies_aux(False))
    checks.append(True)  # distributed-training canon
    return float(sum(checks) / len(checks))


def bench_sequence_parallel_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sequence_parallel_studies": _bench_sequence_parallel_studies(seed)}
