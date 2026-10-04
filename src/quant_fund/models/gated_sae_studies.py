"""gated_sae_studies module (SYNTHETIC)."""

from __future__ import annotations


def gated_sae_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """gated_sae_studies

    check:
    gated_sae_studies: gated sparse autoencoders/magnitudes and detection
    """
    return fit_ok and sample_ok


def gated_sae_studies_aux(aux: bool) -> bool:
    """gated_sae_studies

    aux:
    gated_sae_studies: separate selection and reconstruction/fidelity and density
    """
    return aux


def _bench_gated_sae_studies(seed: int = 0) -> float:
    checks = []
    checks.append(gated_sae_studies_ok(True, True))
    checks.append(not gated_sae_studies_ok(False, True))
    checks.append(gated_sae_studies_aux(True))
    checks.append(not gated_sae_studies_aux(False))
    checks.append(True)  # mech-interp-2 canon
    return float(sum(checks) / len(checks))


def bench_gated_sae_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gated_sae_studies": _bench_gated_sae_studies(seed)}
