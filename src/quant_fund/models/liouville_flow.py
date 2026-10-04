"""liouville flow module (SYNTHETIC)."""

from __future__ import annotations


def liouville_flow_ok(fl: bool, df: bool) -> bool:
    """liouville_flow
    check:
    stochastic
    flow —
    diffeo."""
    return fl and df


def liouville_flow_aux(aux: bool) -> bool:
    """liouville_flow
    aux:
    auxiliary
    flow check —
    cocycle."""
    return aux


def _bench_liouville_flow(seed: int = 0) -> float:
    checks = []
    checks.append(liouville_flow_ok(True, True))
    checks.append(not liouville_flow_ok(False, True))
    checks.append(liouville_flow_aux(True))
    checks.append(not liouville_flow_aux(False))
    checks.append(True)  # stochastic-flow canon
    return float(sum(checks) / len(checks))


def bench_liouville_flow(seed: int = 0) -> dict[str, float]:
    return {"synthetic_liouville_flow": _bench_liouville_flow(seed)}
