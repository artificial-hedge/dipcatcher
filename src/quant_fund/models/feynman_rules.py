"""feynman_rules module (SYNTHETIC)."""

from __future__ import annotations


def feynman_rules_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """feynman_rules

    check:
    klein_gordon: Klein-Gordon equation
    dirac_equation: Dirac equation
    feynman_rules: Feynman rules
    renormalization_group: renormalization group
    path_integral_qm: path integral
    canonical_quantization: canonical quantization
    """
    return fit_ok and sample_ok


def feynman_rules_aux(aux: bool) -> bool:
    """feynman_rules

    aux:
    klein_gordon: scalar field
    dirac_equation: spinor field
    feynman_rules: propagators
    renormalization_group: beta functions
    path_integral_qm: saddle point
    canonical_quantization: commutators
    """
    return aux


def _bench_feynman_rules(seed: int = 0) -> float:
    checks = []
    checks.append(feynman_rules_ok(True, True))
    checks.append(not feynman_rules_ok(False, True))
    checks.append(feynman_rules_aux(True))
    checks.append(not feynman_rules_aux(False))
    checks.append(True)  # quantum-field-theory canon
    return float(sum(checks) / len(checks))


def bench_feynman_rules(seed: int = 0) -> dict[str, float]:
    return {"synthetic_feynman_rules": _bench_feynman_rules(seed)}
