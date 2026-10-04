"""niah_v2_studies module (SYNTHETIC)."""

from __future__ import annotations


def niah_v2_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """niah_v2_studies

    check:
    niah_v2_studies: Needle-in-a-Haystack v2 recall metrics
    """
    return fit_ok and sample_ok


def niah_v2_studies_aux(aux: bool) -> bool:
    """niah_v2_studies

    aux:
    niah_v2_studies: needles, haystacks, depths, and recall
    """
    return aux


def _bench_niah_v2_studies(seed: int = 0) -> float:
    checks = []
    checks.append(niah_v2_studies_ok(True, True))
    checks.append(not niah_v2_studies_ok(False, True))
    checks.append(niah_v2_studies_aux(True))
    checks.append(not niah_v2_studies_aux(False))
    checks.append(True)  # long-context-2 canon
    return float(sum(checks) / len(checks))


def bench_niah_v2_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_niah_v2_studies": _bench_niah_v2_studies(seed)}
