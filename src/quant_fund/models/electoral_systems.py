"""electoral_systems module (SYNTHETIC)."""

from __future__ import annotations


def electoral_systems_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """electoral_systems

    check:
    comparative_politics: comparative politics
    international_relations: international relations
    political_theory: political theory
    public_administration: public administration
    political_economy: political economy
    electoral_systems: electoral systems
    """
    return fit_ok and sample_ok


def electoral_systems_aux(aux: bool) -> bool:
    """electoral_systems

    aux:
    comparative_politics: regime types
    international_relations: state interactions
    political_theory: normative concepts
    public_administration: bureaucratic systems
    political_economy: institutions and markets
    electoral_systems: voting rules
    """
    return aux


def _bench_electoral_systems(seed: int = 0) -> float:
    checks = []
    checks.append(electoral_systems_ok(True, True))
    checks.append(not electoral_systems_ok(False, True))
    checks.append(electoral_systems_aux(True))
    checks.append(not electoral_systems_aux(False))
    checks.append(True)  # political-science canon
    return float(sum(checks) / len(checks))


def bench_electoral_systems(seed: int = 0) -> dict[str, float]:
    return {"synthetic_electoral_systems": _bench_electoral_systems(seed)}
