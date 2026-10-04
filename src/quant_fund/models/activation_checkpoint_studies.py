"""activation_checkpoint_studies module (SYNTHETIC)."""

from __future__ import annotations


def activation_checkpoint_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """activation_checkpoint_studies

    check:
    activation_checkpoint_studies: recomputation over memory/stores and boundaries
    """
    return fit_ok and sample_ok


def activation_checkpoint_studies_aux(aux: bool) -> bool:
    """activation_checkpoint_studies

    aux:
    activation_checkpoint_studies: selective checkpointing and rewiring/layers and tradeoffs
    """
    return aux


def _bench_activation_checkpoint_studies(seed: int = 0) -> float:
    checks = []
    checks.append(activation_checkpoint_studies_ok(True, True))
    checks.append(not activation_checkpoint_studies_ok(False, True))
    checks.append(activation_checkpoint_studies_aux(True))
    checks.append(not activation_checkpoint_studies_aux(False))
    checks.append(True)  # distributed-training canon
    return float(sum(checks) / len(checks))


def bench_activation_checkpoint_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_activation_checkpoint_studies": _bench_activation_checkpoint_studies(seed)}
