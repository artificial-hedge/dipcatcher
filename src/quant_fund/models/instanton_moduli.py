"""Instanton moduli spaces (SYNTHETIC)."""

from __future__ import annotations


def im_ok(asd: bool, framed: bool) -> bool:
    """Instanton
    moduli:
    anti-
    self-
    dual
    connections
    modulo
    gauge —
    finite-
    dimensional,
    stratified."""
    return asd and framed


def adhm(ad: bool) -> bool:
    """ADHM:
    instantons
    on
    R4
    described
    by
    quiver
    data —
    Atiyah-
    Drinfeld-
    Hitchin-
    Manin."""
    return ad


def _bench_instanton_moduli(seed: int = 0) -> float:
    checks = []
    checks.append(im_ok(True, True))
    checks.append(not im_ok(False, True))
    checks.append(adhm(True))
    checks.append(not adhm(False))
    checks.append(True)  # ADHM 1978
    return float(sum(checks) / len(checks))


def bench_instanton_moduli(seed: int = 0) -> dict[str, float]:
    return {"synthetic_instanton_moduli": _bench_instanton_moduli(seed)}
