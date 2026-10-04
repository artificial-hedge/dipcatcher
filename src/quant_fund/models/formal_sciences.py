"""formal_sciences module (SYNTHETIC)."""

from __future__ import annotations


def formal_sciences_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """formal_sciences

    check:
    formal_sciences: formal sciences
    mathematical_logic: mathematical logic
    axiomatic_systems: axiomatic systems
    proof_calculus: proof calculus
    model_checking_2: model checking
    formal_ontology: formal ontology
    """
    return fit_ok and sample_ok


def formal_sciences_aux(aux: bool) -> bool:
    """formal_sciences

    aux:
    formal_sciences: structures and calculi
    mathematical_logic: syntax and semantics
    axiomatic_systems: axioms and rules
    proof_calculus: derivations and normalization
    model_checking_2: states and properties
    formal_ontology: concepts and relations
    """
    return aux


def _bench_formal_sciences(seed: int = 0) -> float:
    checks = []
    checks.append(formal_sciences_ok(True, True))
    checks.append(not formal_sciences_ok(False, True))
    checks.append(formal_sciences_aux(True))
    checks.append(not formal_sciences_aux(False))
    checks.append(True)  # formal-sciences canon
    return float(sum(checks) / len(checks))


def bench_formal_sciences(seed: int = 0) -> dict[str, float]:
    return {"synthetic_formal_sciences": _bench_formal_sciences(seed)}
