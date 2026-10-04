"""bcs_theory module (SYNTHETIC)."""

from __future__ import annotations


def bcs_theory_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """bcs_theory

    check:
    bcs_theory: BCS theory
    nuclear_shell_model: nuclear shell model
    nuclear_liquid_drop: liquid drop model
    quark_model: quark model
    parton_model: parton model
    cabibbo_km: Cabibbo-Kobayashi-Maskawa matrix
    """
    return fit_ok and sample_ok


def bcs_theory_aux(aux: bool) -> bool:
    """bcs_theory

    aux:
    bcs_theory: Cooper pairs
    nuclear_shell_model: magic numbers
    nuclear_liquid_drop: binding energy
    quark_model: baryons
    parton_model: deep inelastic scattering
    cabibbo_km: quark mixing
    """
    return aux


def _bench_bcs_theory(seed: int = 0) -> float:
    checks = []
    checks.append(bcs_theory_ok(True, True))
    checks.append(not bcs_theory_ok(False, True))
    checks.append(bcs_theory_aux(True))
    checks.append(not bcs_theory_aux(False))
    checks.append(True)  # nuclear/particle-physics canon
    return float(sum(checks) / len(checks))


def bench_bcs_theory(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bcs_theory": _bench_bcs_theory(seed)}
