"""cabibbo_km module (SYNTHETIC)."""

from __future__ import annotations


def cabibbo_km_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """cabibbo_km

    check:
    bcs_theory: BCS theory
    nuclear_shell_model: nuclear shell model
    nuclear_liquid_drop: liquid drop model
    quark_model: quark model
    parton_model: parton model
    cabibbo_km: Cabibbo-Kobayashi-Maskawa matrix
    """
    return fit_ok and sample_ok


def cabibbo_km_aux(aux: bool) -> bool:
    """cabibbo_km

    aux:
    bcs_theory: Cooper pairs
    nuclear_shell_model: magic numbers
    nuclear_liquid_drop: binding energy
    quark_model: baryons
    parton_model: deep inelastic scattering
    cabibbo_km: quark mixing
    """
    return aux


def _bench_cabibbo_km(seed: int = 0) -> float:
    checks = []
    checks.append(cabibbo_km_ok(True, True))
    checks.append(not cabibbo_km_ok(False, True))
    checks.append(cabibbo_km_aux(True))
    checks.append(not cabibbo_km_aux(False))
    checks.append(True)  # nuclear/particle-physics canon
    return float(sum(checks) / len(checks))


def bench_cabibbo_km(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cabibbo_km": _bench_cabibbo_km(seed)}
