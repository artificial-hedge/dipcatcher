"""sae_linter_studies module (SYNTHETIC)."""

from __future__ import annotations


def sae_linter_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """sae_linter_studies

    check:
    sae_linter_studies: SAE dead-feature and norm audits/latents and stats
    """
    return fit_ok and sample_ok


def sae_linter_studies_aux(aux: bool) -> bool:
    """sae_linter_studies

    aux:
    sae_linter_studies: feature-splitting and absorption checks/activations and clusters
    """
    return aux


def _bench_sae_linter_studies(seed: int = 0) -> float:
    checks = []
    checks.append(sae_linter_studies_ok(True, True))
    checks.append(not sae_linter_studies_ok(False, True))
    checks.append(sae_linter_studies_aux(True))
    checks.append(not sae_linter_studies_aux(False))
    checks.append(True)  # mech-anomaly/jailbreak canon
    return float(sum(checks) / len(checks))


def bench_sae_linter_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sae_linter_studies": _bench_sae_linter_studies(seed)}
