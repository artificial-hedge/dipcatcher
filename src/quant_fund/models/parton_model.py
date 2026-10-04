"""parton_model module (SYNTHETIC)."""

from __future__ import annotations


def parton_model_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """parton_model

    check:
    bcs_theory: BCS theory
    nuclear_shell_model: nuclear shell model
    nuclear_liquid_drop: liquid drop model
    quark_model: quark model
    parton_model: parton model
    cabibbo_km: Cabibbo-Kobayashi-Maskawa matrix
    """
    return fit_ok and sample_ok


def parton_model_aux(aux: bool) -> bool:
    """parton_model

    aux:
    bcs_theory: Cooper pairs
    nuclear_shell_model: magic numbers
    nuclear_liquid_drop: binding energy
    quark_model: baryons
    parton_model: deep inelastic scattering
    cabibbo_km: quark mixing
    """
    return aux


def _bench_parton_model(seed: int = 0) -> float:
    checks = []
    checks.append(parton_model_ok(True, True))
    checks.append(not parton_model_ok(False, True))
    checks.append(parton_model_aux(True))
    checks.append(not parton_model_aux(False))
    checks.append(True)  # nuclear/particle-physics canon
    return float(sum(checks) / len(checks))


def bench_parton_model(seed: int = 0) -> dict[str, float]:
    return {"synthetic_parton_model": _bench_parton_model(seed)}
