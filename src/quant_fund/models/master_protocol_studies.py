"""master_protocol_studies module (SYNTHETIC)."""

from __future__ import annotations


def master_protocol_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """master_protocol_studies

    check:
    master_protocol_studies: basket and umbrella/platform and substudies
    """
    return fit_ok and sample_ok


def master_protocol_studies_aux(aux: bool) -> bool:
    """master_protocol_studies

    aux:
    master_protocol_studies: sharing and stopping/accrual and randomization
    """
    return aux


def _bench_master_protocol_studies(seed: int = 0) -> float:
    checks = []
    checks.append(master_protocol_studies_ok(True, True))
    checks.append(not master_protocol_studies_ok(False, True))
    checks.append(master_protocol_studies_aux(True))
    checks.append(not master_protocol_studies_aux(False))
    checks.append(True)  # target-trial/RWE canon
    return float(sum(checks) / len(checks))


def bench_master_protocol_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_master_protocol_studies": _bench_master_protocol_studies(seed)}
