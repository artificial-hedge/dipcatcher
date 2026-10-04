"""occupancy_measure_studies module (SYNTHETIC)."""

from __future__ import annotations


def occupancy_measure_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """occupancy_measure_studies

    check:
    occupancy_measure_studies: state visitation and stationary distributions/matching and MaxEnt
    """
    return fit_ok and sample_ok


def occupancy_measure_studies_aux(aux: bool) -> bool:
    """occupancy_measure_studies

    aux:
    occupancy_measure_studies: dual and primal methods/distributions and covering
    """
    return aux


def _bench_occupancy_measure_studies(seed: int = 0) -> float:
    checks = []
    checks.append(occupancy_measure_studies_ok(True, True))
    checks.append(not occupancy_measure_studies_ok(False, True))
    checks.append(occupancy_measure_studies_aux(True))
    checks.append(not occupancy_measure_studies_aux(False))
    checks.append(True)  # RL-skills/goal canon
    return float(sum(checks) / len(checks))


def bench_occupancy_measure_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_occupancy_measure_studies": _bench_occupancy_measure_studies(seed)}
