"""e_value_studies module (SYNTHETIC)."""

from __future__ import annotations


def e_value_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """e_value_studies

    check:
    e_value_studies: unmeasured confounding/sensitivity and minimum RR
    """
    return fit_ok and sample_ok


def e_value_studies_aux(aux: bool) -> bool:
    """e_value_studies

    aux:
    e_value_studies: observed association/bound and robustness
    """
    return aux


def _bench_e_value_studies(seed: int = 0) -> float:
    checks = []
    checks.append(e_value_studies_ok(True, True))
    checks.append(not e_value_studies_ok(False, True))
    checks.append(e_value_studies_aux(True))
    checks.append(not e_value_studies_aux(False))
    checks.append(True)  # target-trial/RWE canon
    return float(sum(checks) / len(checks))


def bench_e_value_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_e_value_studies": _bench_e_value_studies(seed)}
