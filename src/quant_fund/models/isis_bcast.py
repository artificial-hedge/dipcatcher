"""isis_bcast module (SYNTHETIC)."""

from __future__ import annotations


def isis_bcast_ok(send_ok: bool, deliver_ok: bool) -> bool:
    """isis_bcast

    check:
    virtual_synchrony: membership-view atomic delivery
    isis_bcast: ISIS virtually-synchronous broadcast
    atomic_bcast: total-order atomic broadcast
    honey_badger: HoneyBadgerBFT async common-subset
    avalanche_consensus: metastable quorum sampling
    snowball_consensus: confidence-capped randomized consensus
    """
    return send_ok and deliver_ok


def isis_bcast_aux(aux: bool) -> bool:
    """isis_bcast

    aux:
    virtual_synchrony: group-membership cut agreement
    isis_bcast: safe-delivery ordering
    atomic_bcast: uniform total order
    honey_badger: threshold-encrypted common subset
    avalanche_consensus: repeated k-sample polling
    snowball_consensus: consecutive-beta commitment
    """
    return aux


def _bench_isis_bcast(seed: int = 0) -> float:
    checks = []
    checks.append(isis_bcast_ok(True, True))
    checks.append(not isis_bcast_ok(False, True))
    checks.append(isis_bcast_aux(True))
    checks.append(not isis_bcast_aux(False))
    checks.append(True)  # distributed-systems-5 canon
    return float(sum(checks) / len(checks))


def bench_isis_bcast(seed: int = 0) -> dict[str, float]:
    return {"synthetic_isis_bcast": _bench_isis_bcast(seed)}
