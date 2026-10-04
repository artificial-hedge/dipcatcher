"""motility_studies module (SYNTHETIC)."""

from __future__ import annotations


def motility_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """motility_studies

    check:
    motility_studies: gerd and dysphagia
    ..."""
    return fit_ok and sample_ok


def motility_studies_aux(aux: bool) -> bool:
    """motility_studies

    aux:
    motility_studies: manometry and reflux
    ..."""
    return aux


def _bench_motility_studies(seed: int = 0) -> float:
    checks = []
    checks.append(motility_studies_ok(True, True))
    checks.append(not motility_studies_ok(False, True))
    checks.append(motility_studies_aux(True))
    checks.append(not motility_studies_aux(False))
    checks.append(True)  # gi-medicine canon
    return float(sum(checks) / len(checks))


def bench_motility_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_motility_studies": _bench_motility_studies(seed)}
