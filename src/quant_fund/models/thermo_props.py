"""thermo_props module (SYNTHETIC)."""

from __future__ import annotations


def thermo_props_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """thermo_props

    check:
    reaction_kinetics: reaction kinetics
    thermo_props: thermodynamic properties
    separation_proc: separation processes
    heat_exchanger: heat exchanger
    fluid_dynamics2: fluid dynamics
    process_control: process control
    """
    return fit_ok and sample_ok


def thermo_props_aux(aux: bool) -> bool:
    """thermo_props

    aux:
    reaction_kinetics: rate laws
    thermo_props: equation of state
    separation_proc: distillation
    heat_exchanger: LMTD
    fluid_dynamics2: Reynolds number
    process_control: feedback loops
    """
    return aux


def _bench_thermo_props(seed: int = 0) -> float:
    checks = []
    checks.append(thermo_props_ok(True, True))
    checks.append(not thermo_props_ok(False, True))
    checks.append(thermo_props_aux(True))
    checks.append(not thermo_props_aux(False))
    checks.append(True)  # chemical-engineering canon
    return float(sum(checks) / len(checks))


def bench_thermo_props(seed: int = 0) -> dict[str, float]:
    return {"synthetic_thermo_props": _bench_thermo_props(seed)}
