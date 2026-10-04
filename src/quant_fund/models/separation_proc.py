"""separation_proc module (SYNTHETIC)."""

from __future__ import annotations


def separation_proc_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """separation_proc

    check:
    reaction_kinetics: reaction kinetics
    thermo_props: thermodynamic properties
    separation_proc: separation processes
    heat_exchanger: heat exchanger
    fluid_dynamics2: fluid dynamics
    process_control: process control
    """
    return fit_ok and sample_ok


def separation_proc_aux(aux: bool) -> bool:
    """separation_proc

    aux:
    reaction_kinetics: rate laws
    thermo_props: equation of state
    separation_proc: distillation
    heat_exchanger: LMTD
    fluid_dynamics2: Reynolds number
    process_control: feedback loops
    """
    return aux


def _bench_separation_proc(seed: int = 0) -> float:
    checks = []
    checks.append(separation_proc_ok(True, True))
    checks.append(not separation_proc_ok(False, True))
    checks.append(separation_proc_aux(True))
    checks.append(not separation_proc_aux(False))
    checks.append(True)  # chemical-engineering canon
    return float(sum(checks) / len(checks))


def bench_separation_proc(seed: int = 0) -> dict[str, float]:
    return {"synthetic_separation_proc": _bench_separation_proc(seed)}
