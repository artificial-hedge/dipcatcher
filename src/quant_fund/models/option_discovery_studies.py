"""option_discovery_studies module (SYNTHETIC)."""

from __future__ import annotations


def option_discovery_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """option_discovery_studies

    check:
    option_discovery_studies: options and termination/subgoals and eigenoptions
    """
    return fit_ok and sample_ok


def option_discovery_studies_aux(aux: bool) -> bool:
    """option_discovery_studies

    aux:
    option_discovery_studies: bottlenecks and covering/options-critic and hierarchies
    """
    return aux


def _bench_option_discovery_studies(seed: int = 0) -> float:
    checks = []
    checks.append(option_discovery_studies_ok(True, True))
    checks.append(not option_discovery_studies_ok(False, True))
    checks.append(option_discovery_studies_aux(True))
    checks.append(not option_discovery_studies_aux(False))
    checks.append(True)  # RL-skills/goal canon
    return float(sum(checks) / len(checks))


def bench_option_discovery_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_option_discovery_studies": _bench_option_discovery_studies(seed)}
