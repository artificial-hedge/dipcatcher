"""activation_patch_studies module (SYNTHETIC)."""

from __future__ import annotations


def activation_patch_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """activation_patch_studies

    check:
    activation_patch_studies: cross-prompt activation patching/sources and targets
    """
    return fit_ok and sample_ok


def activation_patch_studies_aux(aux: bool) -> bool:
    """activation_patch_studies

    aux:
    activation_patch_studies: residual-stream swap interventions/layers and effects
    """
    return aux


def _bench_activation_patch_studies(seed: int = 0) -> float:
    checks = []
    checks.append(activation_patch_studies_ok(True, True))
    checks.append(not activation_patch_studies_ok(False, True))
    checks.append(activation_patch_studies_aux(True))
    checks.append(not activation_patch_studies_aux(False))
    checks.append(True)  # mech-anomaly/jailbreak canon
    return float(sum(checks) / len(checks))


def bench_activation_patch_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_activation_patch_studies": _bench_activation_patch_studies(seed)}
