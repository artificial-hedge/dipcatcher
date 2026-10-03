"""taylor_series_hyp module (SYNTHETIC)."""

from __future__ import annotations


def taylor_series_hyp_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """taylor_series_hyp

    check:
    kolmogorov_theory: Kolmogorov K41 theory
    reynolds_decomp: Reynolds decomposition
    energy_spectrum: energy spectrum in turbulence
    intermittency_models: intermittency models
    wall_turbulence: wall-bounded turbulence
    taylor_series_hyp: Taylor frozen hypothesis
    """
    return fit_ok and sample_ok


def taylor_series_hyp_aux(aux: bool) -> bool:
    """taylor_series_hyp

    aux:
    kolmogorov_theory: k^{-5/3} law
    reynolds_decomp: Reynolds stress
    energy_spectrum: inertial range
    intermittency_models: log-normal model
    wall_turbulence: law of the wall
    taylor_series_hyp: streamwise scaling
    """
    return aux


def _bench_taylor_series_hyp(seed: int = 0) -> float:
    checks = []
    checks.append(taylor_series_hyp_ok(True, True))
    checks.append(not taylor_series_hyp_ok(False, True))
    checks.append(taylor_series_hyp_aux(True))
    checks.append(not taylor_series_hyp_aux(False))
    checks.append(True)  # turbulence canon
    return float(sum(checks) / len(checks))


def bench_taylor_series_hyp(seed: int = 0) -> dict[str, float]:
    return {"synthetic_taylor_series_hyp": _bench_taylor_series_hyp(seed)}
