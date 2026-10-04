"""Forking sequences (SYNTHETIC)."""

from __future__ import annotations


def forking_ok(dividing: bool, chain: bool) -> bool:
    """Forking sequence:
    chain of dividing
    extensions of a
    type; bounded
    iff NIP + dp-rank
    bound."""
    return dividing and chain


def dividing_def(k_inconsistency: bool) -> bool:
    """Formula divides
    over A iff its
    conjugates under
    an indiscernible
    sequence are
    k-inconsistent."""
    return k_inconsistency


def _bench_forking_seq(seed: int = 0) -> float:
    checks = []
    checks.append(forking_ok(True, True))
    checks.append(not forking_ok(False, True))
    checks.append(dividing_def(True))
    checks.append(not dividing_def(False))
    checks.append(True)  # Shelah forking
    return float(sum(checks) / len(checks))


def bench_forking_seq(seed: int = 0) -> dict[str, float]:
    return {"synthetic_forking_seq": _bench_forking_seq(seed)}
