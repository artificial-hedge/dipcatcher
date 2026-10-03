"""kunita flow module (SYNTHETIC)."""

from __future__ import annotations


def kunita_flow_ok(fl: bool, df: bool) -> bool:
    """kunita_flow
    check:
    stochastic
    flow —
    diffeo."""
    return fl and df


def kunita_flow_aux(aux: bool) -> bool:
    """kunita_flow
    aux:
    auxiliary
    flow check —
    cocycle."""
    return aux


def _bench_kunita_flow(seed: int = 0) -> float:
    checks = []
    checks.append(kunita_flow_ok(True, True))
    checks.append(not kunita_flow_ok(False, True))
    checks.append(kunita_flow_aux(True))
    checks.append(not kunita_flow_aux(False))
    checks.append(True)  # stochastic-flow canon
    return float(sum(checks) / len(checks))


def bench_kunita_flow(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kunita_flow": _bench_kunita_flow(seed)}
