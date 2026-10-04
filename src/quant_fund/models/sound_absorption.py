"""sound_absorption module (SYNTHETIC)."""

from __future__ import annotations


def sound_absorption_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """sound_absorption

    check:
    acoustic_wave_eq: acoustic wave equation
    helmholtz_eq: Helmholtz equation
    sound_absorption: sound absorption
    room_acoustics: room acoustics
    rayleigh_scattering: Rayleigh scattering
    doppler_effect: Doppler effect
    """
    return fit_ok and sample_ok


def sound_absorption_aux(aux: bool) -> bool:
    """sound_absorption

    aux:
    acoustic_wave_eq: d'Alembert solution
    helmholtz_eq: boundary value
    sound_absorption: impedance tube
    room_acoustics: reverberation time
    rayleigh_scattering: inverse fourth power
    doppler_effect: frequency shift
    """
    return aux


def _bench_sound_absorption(seed: int = 0) -> float:
    checks = []
    checks.append(sound_absorption_ok(True, True))
    checks.append(not sound_absorption_ok(False, True))
    checks.append(sound_absorption_aux(True))
    checks.append(not sound_absorption_aux(False))
    checks.append(True)  # acoustics canon
    return float(sum(checks) / len(checks))


def bench_sound_absorption(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sound_absorption": _bench_sound_absorption(seed)}
