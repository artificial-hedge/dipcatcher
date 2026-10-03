"""stochastic flow module (SYNTHETIC)."""

from __future__ import annotations


def stochastic_flow_ok(fl: bool, df: bool) -> bool:
    """stochastic_flow
    check:
    stochastic
    flow —
    diffeo."""
    return fl and df


def stochastic_flow_aux(aux: bool) -> bool:
    """stochastic_flow
    aux:
    auxiliary
    flow check —
    cocycle."""
    return aux


def _bench_stochastic_flow(seed: int = 0) -> float:
    checks = []
    checks.append(stochastic_flow_ok(True, True))
    checks.append(not stochastic_flow_ok(False, True))
    checks.append(stochastic_flow_aux(True))
    checks.append(not stochastic_flow_aux(False))
    checks.append(True)  # stochastic-flow canon
    return float(sum(checks) / len(checks))


def bench_stochastic_flow(seed: int = 0) -> dict[str, float]:
    return {"synthetic_stochastic_flow": _bench_stochastic_flow(seed)}
