"""strip_defense_studies module (SYNTHETIC)."""

from __future__ import annotations


def strip_defense_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """strip_defense_studies

    check:
    strip_defense_studies: STRIP input-entropy backdoor screening and rates
    """
    return fit_ok and sample_ok


def strip_defense_studies_aux(aux: bool) -> bool:
    """strip_defense_studies

    aux:
    strip_defense_studies: perturbed inputs, entropy tails, and detection
    """
    return aux


def _bench_strip_defense_studies(seed: int = 0) -> float:
    checks = []
    checks.append(strip_defense_studies_ok(True, True))
    checks.append(not strip_defense_studies_ok(False, True))
    checks.append(strip_defense_studies_aux(True))
    checks.append(not strip_defense_studies_aux(False))
    checks.append(True)  # privacy-attack canon
    return float(sum(checks) / len(checks))


def bench_strip_defense_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_strip_defense_studies": _bench_strip_defense_studies(seed)}
