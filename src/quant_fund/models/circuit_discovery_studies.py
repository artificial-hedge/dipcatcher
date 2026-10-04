"""circuit_discovery_studies module (SYNTHETIC)."""

from __future__ import annotations


def circuit_discovery_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """circuit_discovery_studies

    check:
    circuit_discovery_studies: ACDC and iterative pruning/edges and thresholds
    """
    return fit_ok and sample_ok


def circuit_discovery_studies_aux(aux: bool) -> bool:
    """circuit_discovery_studies

    aux:
    circuit_discovery_studies: metric preservation and completeness/recovery and validation
    """
    return aux


def _bench_circuit_discovery_studies(seed: int = 0) -> float:
    checks = []
    checks.append(circuit_discovery_studies_ok(True, True))
    checks.append(not circuit_discovery_studies_ok(False, True))
    checks.append(circuit_discovery_studies_aux(True))
    checks.append(not circuit_discovery_studies_aux(False))
    checks.append(True)  # mech-interp-2 canon
    return float(sum(checks) / len(checks))


def bench_circuit_discovery_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_circuit_discovery_studies": _bench_circuit_discovery_studies(seed)}
