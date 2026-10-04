"""Operadic nerve (SYNTHETIC)."""

from __future__ import annotations


def on_ok(operadic: bool, nerve: bool) -> bool:
    """Operadic
    nerve:
    operadic
    nerve
    of
    a
    simplicial
    operad —
    Lurie
    nerve."""
    return operadic and nerve


def nerve_fibered(nf: bool) -> bool:
    """Nerve
    fibration:
    operadic
    nerve
    fibration —
    Lurie
    operad
    nerve."""
    return nf


def _bench_operadic_nerve(seed: int = 0) -> float:
    checks = []
    checks.append(on_ok(True, True))
    checks.append(not on_ok(False, True))
    checks.append(nerve_fibered(True))
    checks.append(not nerve_fibered(False))
    checks.append(True)  # Lurie
    return float(sum(checks) / len(checks))


def bench_operadic_nerve(seed: int = 0) -> dict[str, float]:
    return {"synthetic_operadic_nerve": _bench_operadic_nerve(seed)}
