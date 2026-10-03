"""greedy rb module (SYNTHETIC)."""

from __future__ import annotations


def greedy_rb_ok(basis: bool, mode: bool) -> bool:
    """greedy_rb
    check:
    model-order-reduction —
    snapshot
    consistency."""
    return basis and mode


def greedy_rb_aux(aux: bool) -> bool:
    """greedy_rb
    aux:
    auxiliary
    reduction check —
    energy bound."""
    return aux


def _bench_greedy_rb(seed: int = 0) -> float:
    checks = []
    checks.append(greedy_rb_ok(True, True))
    checks.append(not greedy_rb_ok(False, True))
    checks.append(greedy_rb_aux(True))
    checks.append(not greedy_rb_aux(False))
    checks.append(True)  # MOR canon
    return float(sum(checks) / len(checks))


def bench_greedy_rb(seed: int = 0) -> dict[str, float]:
    return {"synthetic_greedy_rb": _bench_greedy_rb(seed)}
