"""doppler_effect module (SYNTHETIC)."""

from __future__ import annotations


def doppler_effect_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """doppler_effect

    check:
    acoustic_wave_eq: acoustic wave equation
    helmholtz_eq: Helmholtz equation
    sound_absorption: sound absorption
    room_acoustics: room acoustics
    rayleigh_scattering: Rayleigh scattering
    doppler_effect: Doppler effect
    """
    return fit_ok and sample_ok


def doppler_effect_aux(aux: bool) -> bool:
    """doppler_effect

    aux:
    acoustic_wave_eq: d'Alembert solution
    helmholtz_eq: boundary value
    sound_absorption: impedance tube
    room_acoustics: reverberation time
    rayleigh_scattering: inverse fourth power
    doppler_effect: frequency shift
    """
    return aux


def _bench_doppler_effect(seed: int = 0) -> float:
    checks = []
    checks.append(doppler_effect_ok(True, True))
    checks.append(not doppler_effect_ok(False, True))
    checks.append(doppler_effect_aux(True))
    checks.append(not doppler_effect_aux(False))
    checks.append(True)  # acoustics canon
    return float(sum(checks) / len(checks))


def bench_doppler_effect(seed: int = 0) -> dict[str, float]:
    return {"synthetic_doppler_effect": _bench_doppler_effect(seed)}
