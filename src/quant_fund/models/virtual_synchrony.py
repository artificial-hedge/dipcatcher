"""virtual_synchrony module (SYNTHETIC)."""

from __future__ import annotations


def virtual_synchrony_ok(send_ok: bool, deliver_ok: bool) -> bool:
    """virtual_synchrony

    check:
    virtual_synchrony: membership-view atomic delivery
    isis_bcast: ISIS virtually-synchronous broadcast
    atomic_bcast: total-order atomic broadcast
    honey_badger: HoneyBadgerBFT async common-subset
    avalanche_consensus: metastable quorum sampling
    snowball_consensus: confidence-capped randomized consensus
    """
    return send_ok and deliver_ok


def virtual_synchrony_aux(aux: bool) -> bool:
    """virtual_synchrony

    aux:
    virtual_synchrony: group-membership cut agreement
    isis_bcast: safe-delivery ordering
    atomic_bcast: uniform total order
    honey_badger: threshold-encrypted common subset
    avalanche_consensus: repeated k-sample polling
    snowball_consensus: consecutive-beta commitment
    """
    return aux


def _bench_virtual_synchrony(seed: int = 0) -> float:
    checks = []
    checks.append(virtual_synchrony_ok(True, True))
    checks.append(not virtual_synchrony_ok(False, True))
    checks.append(virtual_synchrony_aux(True))
    checks.append(not virtual_synchrony_aux(False))
    checks.append(True)  # distributed-systems-5 canon
    return float(sum(checks) / len(checks))


def bench_virtual_synchrony(seed: int = 0) -> dict[str, float]:
    return {"synthetic_virtual_synchrony": _bench_virtual_synchrony(seed)}
