"""nuclear_liquid_drop module (SYNTHETIC)."""

from __future__ import annotations


def nuclear_liquid_drop_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """nuclear_liquid_drop

    check:
    bcs_theory: BCS theory
    nuclear_shell_model: nuclear shell model
    nuclear_liquid_drop: liquid drop model
    quark_model: quark model
    parton_model: parton model
    cabibbo_km: Cabibbo-Kobayashi-Maskawa matrix
    """
    return fit_ok and sample_ok


def nuclear_liquid_drop_aux(aux: bool) -> bool:
    """nuclear_liquid_drop

    aux:
    bcs_theory: Cooper pairs
    nuclear_shell_model: magic numbers
    nuclear_liquid_drop: binding energy
    quark_model: baryons
    parton_model: deep inelastic scattering
    cabibbo_km: quark mixing
    """
    return aux


def _bench_nuclear_liquid_drop(seed: int = 0) -> float:
    checks = []
    checks.append(nuclear_liquid_drop_ok(True, True))
    checks.append(not nuclear_liquid_drop_ok(False, True))
    checks.append(nuclear_liquid_drop_aux(True))
    checks.append(not nuclear_liquid_drop_aux(False))
    checks.append(True)  # nuclear/particle-physics canon
    return float(sum(checks) / len(checks))


def bench_nuclear_liquid_drop(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nuclear_liquid_drop": _bench_nuclear_liquid_drop(seed)}
