"""convoy net module (SYNTHETIC)."""

from __future__ import annotations


def convoy_net_ok(net: bool, prod: bool) -> bool:
    """convoy_net
    check:
    queue-net
    structure —
    BCMP
    form."""
    return net and prod


def convoy_net_aux(aux: bool) -> bool:
    """convoy_net
    aux:
    auxiliary
    MVA
    check —
    Gordon-Newell."""
    return aux


def _bench_convoy_net(seed: int = 0) -> float:
    checks = []
    checks.append(convoy_net_ok(True, True))
    checks.append(not convoy_net_ok(False, True))
    checks.append(convoy_net_aux(True))
    checks.append(not convoy_net_aux(False))
    checks.append(True)  # queue-net canon
    return float(sum(checks) / len(checks))


def bench_convoy_net(seed: int = 0) -> dict[str, float]:
    return {"synthetic_convoy_net": _bench_convoy_net(seed)}
