"""fine_pruning_studies module (SYNTHETIC)."""

from __future__ import annotations


def fine_pruning_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """fine_pruning_studies

    check:
    fine_pruning_studies: Fine-Pruning dormant-neuron backdoor removal
    """
    return fit_ok and sample_ok


def fine_pruning_studies_aux(aux: bool) -> bool:
    """fine_pruning_studies

    aux:
    fine_pruning_studies: prune curves, clean acc, and attack success
    """
    return aux


def _bench_fine_pruning_studies(seed: int = 0) -> float:
    checks = []
    checks.append(fine_pruning_studies_ok(True, True))
    checks.append(not fine_pruning_studies_ok(False, True))
    checks.append(fine_pruning_studies_aux(True))
    checks.append(not fine_pruning_studies_aux(False))
    checks.append(True)  # privacy-attack canon
    return float(sum(checks) / len(checks))


def bench_fine_pruning_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fine_pruning_studies": _bench_fine_pruning_studies(seed)}
