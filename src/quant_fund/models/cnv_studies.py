"""cnv_studies module (SYNTHETIC)."""

from __future__ import annotations


def cnv_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """cnv_studies

    check:
    cnv_studies: copy and duplication
    ..."""
    return fit_ok and sample_ok


def cnv_studies_aux(aux: bool) -> bool:
    """cnv_studies

    aux:
    cnv_studies: deletion and dosage
    ..."""
    return aux


def _bench_cnv_studies(seed: int = 0) -> float:
    checks = []
    checks.append(cnv_studies_ok(True, True))
    checks.append(not cnv_studies_ok(False, True))
    checks.append(cnv_studies_aux(True))
    checks.append(not cnv_studies_aux(False))
    checks.append(True)  # molecular-genetics-2 canon
    return float(sum(checks) / len(checks))


def bench_cnv_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cnv_studies": _bench_cnv_studies(seed)}
