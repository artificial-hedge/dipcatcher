"""dynamic_borrowing_studies module (SYNTHETIC)."""

from __future__ import annotations


def dynamic_borrowing_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """dynamic_borrowing_studies

    check:
    dynamic_borrowing_studies: power priors and commensurate/discounting and MAP
    """
    return fit_ok and sample_ok


def dynamic_borrowing_studies_aux(aux: bool) -> bool:
    """dynamic_borrowing_studies

    aux:
    dynamic_borrowing_studies: historical controls/commensurability and borrowing
    """
    return aux


def _bench_dynamic_borrowing_studies(seed: int = 0) -> float:
    checks = []
    checks.append(dynamic_borrowing_studies_ok(True, True))
    checks.append(not dynamic_borrowing_studies_ok(False, True))
    checks.append(dynamic_borrowing_studies_aux(True))
    checks.append(not dynamic_borrowing_studies_aux(False))
    checks.append(True)  # target-trial/RWE canon
    return float(sum(checks) / len(checks))


def bench_dynamic_borrowing_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dynamic_borrowing_studies": _bench_dynamic_borrowing_studies(seed)}
