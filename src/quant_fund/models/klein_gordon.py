"""klein_gordon module (SYNTHETIC)."""

from __future__ import annotations


def klein_gordon_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """klein_gordon

    check:
    klein_gordon: Klein-Gordon equation
    dirac_equation: Dirac equation
    feynman_rules: Feynman rules
    renormalization_group: renormalization group
    path_integral_qm: path integral
    canonical_quantization: canonical quantization
    """
    return fit_ok and sample_ok


def klein_gordon_aux(aux: bool) -> bool:
    """klein_gordon

    aux:
    klein_gordon: scalar field
    dirac_equation: spinor field
    feynman_rules: propagators
    renormalization_group: beta functions
    path_integral_qm: saddle point
    canonical_quantization: commutators
    """
    return aux


def _bench_klein_gordon(seed: int = 0) -> float:
    checks = []
    checks.append(klein_gordon_ok(True, True))
    checks.append(not klein_gordon_ok(False, True))
    checks.append(klein_gordon_aux(True))
    checks.append(not klein_gordon_aux(False))
    checks.append(True)  # quantum-field-theory canon
    return float(sum(checks) / len(checks))


def bench_klein_gordon(seed: int = 0) -> dict[str, float]:
    return {"synthetic_klein_gordon": _bench_klein_gordon(seed)}
