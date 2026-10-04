"""drop_studies module (SYNTHETIC)."""

from __future__ import annotations


def drop_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """drop_studies

    check:
    drop_studies: DROP discrete-reasoning passages and F1/EM
    """
    return fit_ok and sample_ok


def drop_studies_aux(aux: bool) -> bool:
    """drop_studies

    aux:
    drop_studies: numerical/spatial reasoning, counts, answers
    """
    return aux


def _bench_drop_studies(seed: int = 0) -> float:
    checks = []
    checks.append(drop_studies_ok(True, True))
    checks.append(not drop_studies_ok(False, True))
    checks.append(drop_studies_aux(True))
    checks.append(not drop_studies_aux(False))
    checks.append(True)  # reading-comprehension canon
    return float(sum(checks) / len(checks))


def bench_drop_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_drop_studies": _bench_drop_studies(seed)}
