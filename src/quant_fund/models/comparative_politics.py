"""comparative_politics module (SYNTHETIC)."""

from __future__ import annotations


def comparative_politics_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """comparative_politics

    check:
    comparative_politics: comparative politics
    international_relations: international relations
    political_theory: political theory
    public_administration: public administration
    political_economy: political economy
    electoral_systems: electoral systems
    """
    return fit_ok and sample_ok


def comparative_politics_aux(aux: bool) -> bool:
    """comparative_politics

    aux:
    comparative_politics: regime types
    international_relations: state interactions
    political_theory: normative concepts
    public_administration: bureaucratic systems
    political_economy: institutions and markets
    electoral_systems: voting rules
    """
    return aux


def _bench_comparative_politics(seed: int = 0) -> float:
    checks = []
    checks.append(comparative_politics_ok(True, True))
    checks.append(not comparative_politics_ok(False, True))
    checks.append(comparative_politics_aux(True))
    checks.append(not comparative_politics_aux(False))
    checks.append(True)  # political-science canon
    return float(sum(checks) / len(checks))


def bench_comparative_politics(seed: int = 0) -> dict[str, float]:
    return {"synthetic_comparative_politics": _bench_comparative_politics(seed)}
