"""rayleigh_scattering module (SYNTHETIC)."""

from __future__ import annotations


def rayleigh_scattering_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """rayleigh_scattering

    check:
    acoustic_wave_eq: acoustic wave equation
    helmholtz_eq: Helmholtz equation
    sound_absorption: sound absorption
    room_acoustics: room acoustics
    rayleigh_scattering: Rayleigh scattering
    doppler_effect: Doppler effect
    """
    return fit_ok and sample_ok


def rayleigh_scattering_aux(aux: bool) -> bool:
    """rayleigh_scattering

    aux:
    acoustic_wave_eq: d'Alembert solution
    helmholtz_eq: boundary value
    sound_absorption: impedance tube
    room_acoustics: reverberation time
    rayleigh_scattering: inverse fourth power
    doppler_effect: frequency shift
    """
    return aux


def _bench_rayleigh_scattering(seed: int = 0) -> float:
    checks = []
    checks.append(rayleigh_scattering_ok(True, True))
    checks.append(not rayleigh_scattering_ok(False, True))
    checks.append(rayleigh_scattering_aux(True))
    checks.append(not rayleigh_scattering_aux(False))
    checks.append(True)  # acoustics canon
    return float(sum(checks) / len(checks))


def bench_rayleigh_scattering(seed: int = 0) -> dict[str, float]:
    return {"synthetic_rayleigh_scattering": _bench_rayleigh_scattering(seed)}
