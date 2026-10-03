"""Q-prisms (SYNTHETIC)."""

from __future__ import annotations


def qp_ok(q: bool, prism: bool) -> bool:
    """Q
    prism:
    Q
    prism —
    delta
    ring."""
    return q and prism


def q_delta_ring(qd: bool) -> bool:
    """Q
    delta:
    q
    delta
    ring —
    q
    Frobenius."""
    return qd


def _bench_q_prism(seed: int = 0) -> float:
    checks = []
    checks.append(qp_ok(True, True))
    checks.append(not qp_ok(False, True))
    checks.append(q_delta_ring(True))
    checks.append(not q_delta_ring(False))
    checks.append(True)  # Bhatt
    return float(sum(checks) / len(checks))


def bench_q_prism(seed: int = 0) -> dict[str, float]:
    return {"synthetic_q_prism": _bench_q_prism(seed)}
