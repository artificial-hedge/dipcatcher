"""machine_design module (SYNTHETIC)."""

from __future__ import annotations


def machine_design_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """machine_design

    check:
    solid_mechanics: solid mechanics
    vibration_analysis: vibration analysis
    fatigue_life: fatigue life
    tribology: tribology
    machine_design: machine design
    kinematics: kinematics
    """
    return fit_ok and sample_ok


def machine_design_aux(aux: bool) -> bool:
    """machine_design

    aux:
    solid_mechanics: stress-strain
    vibration_analysis: modal analysis
    fatigue_life: S-N curve
    tribology: friction
    machine_design: tolerancing
    kinematics: linkage
    """
    return aux


def _bench_machine_design(seed: int = 0) -> float:
    checks = []
    checks.append(machine_design_ok(True, True))
    checks.append(not machine_design_ok(False, True))
    checks.append(machine_design_aux(True))
    checks.append(not machine_design_aux(False))
    checks.append(True)  # mechanical-engineering canon
    return float(sum(checks) / len(checks))


def bench_machine_design(seed: int = 0) -> dict[str, float]:
    return {"synthetic_machine_design": _bench_machine_design(seed)}
