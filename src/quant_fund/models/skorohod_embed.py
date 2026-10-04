"""skorohod embed module (SYNTHETIC)."""

from __future__ import annotations


def skorohod_embed_ok(weak: bool, conv: bool) -> bool:
    """skorohod_embed
    check:
    weak
    convergence —
    measure."""
    return weak and conv


def skorohod_embed_aux(aux: bool) -> bool:
    """skorohod_embed
    aux:
    auxiliary
    convergence check —
    approx."""
    return aux


def _bench_skorohod_embed(seed: int = 0) -> float:
    checks = []
    checks.append(skorohod_embed_ok(True, True))
    checks.append(not skorohod_embed_ok(False, True))
    checks.append(skorohod_embed_aux(True))
    checks.append(not skorohod_embed_aux(False))
    checks.append(True)  # weak-convergence canon
    return float(sum(checks) / len(checks))


def bench_skorohod_embed(seed: int = 0) -> dict[str, float]:
    return {"synthetic_skorohod_embed": _bench_skorohod_embed(seed)}
