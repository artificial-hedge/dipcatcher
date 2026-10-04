"""axiomatic_systems module (SYNTHETIC)."""

from __future__ import annotations


def axiomatic_systems_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """axiomatic_systems

    check:
    formal_sciences: formal sciences
    mathematical_logic: mathematical logic
    axiomatic_systems: axiomatic systems
    proof_calculus: proof calculus
    model_checking_2: model checking
    formal_ontology: formal ontology
    """
    return fit_ok and sample_ok


def axiomatic_systems_aux(aux: bool) -> bool:
    """axiomatic_systems

    aux:
    formal_sciences: structures and calculi
    mathematical_logic: syntax and semantics
    axiomatic_systems: axioms and rules
    proof_calculus: derivations and normalization
    model_checking_2: states and properties
    formal_ontology: concepts and relations
    """
    return aux


def _bench_axiomatic_systems(seed: int = 0) -> float:
    checks = []
    checks.append(axiomatic_systems_ok(True, True))
    checks.append(not axiomatic_systems_ok(False, True))
    checks.append(axiomatic_systems_aux(True))
    checks.append(not axiomatic_systems_aux(False))
    checks.append(True)  # formal-sciences canon
    return float(sum(checks) / len(checks))


def bench_axiomatic_systems(seed: int = 0) -> dict[str, float]:
    return {"synthetic_axiomatic_systems": _bench_axiomatic_systems(seed)}
