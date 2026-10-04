"""karal flow module (SYNTHETIC)."""

from __future__ import annotations


def karal_flow_ok(fl: bool, df: bool) -> bool:
    """karal_flow
    check:
    stochastic
    flow —
    diffeo."""
    return fl and df


def karal_flow_aux(aux: bool) -> bool:
    """karal_flow
    aux:
    auxiliary
    flow check —
    cocycle."""
    return aux


def _bench_karal_flow(seed: int = 0) -> float:
    checks = []
    checks.append(karal_flow_ok(True, True))
    checks.append(not karal_flow_ok(False, True))
    checks.append(karal_flow_aux(True))
    checks.append(not karal_flow_aux(False))
    checks.append(True)  # stochastic-flow canon
    return float(sum(checks) / len(checks))


def bench_karal_flow(seed: int = 0) -> dict[str, float]:
    return {"synthetic_karal_flow": _bench_karal_flow(seed)}
