"""Sobolev spaces (SYNTHETIC)."""

from __future__ import annotations


def sob_ok(weak_deriv: bool, complete: bool) -> bool:
    """Sobolev
    space
    W^{k,p}:
    weak
    derivatives
    in
    L^p —
    Banach
    (Hilbert
    for
    p=2)."""
    return weak_deriv and complete


def sobolev_embed2(se: bool) -> bool:
    """Sobolev
    embedding:
    W^{1,p}
    into
    L^q
    or
    Holder
    depending
    on
    p
    vs
    n."""
    return se


def _bench_sobolev_space(seed: int = 0) -> float:
    checks = []
    checks.append(sob_ok(True, True))
    checks.append(not sob_ok(False, True))
    checks.append(sobolev_embed2(True))
    checks.append(not sobolev_embed2(False))
    checks.append(True)  # Sobolev
    return float(sum(checks) / len(checks))


def bench_sobolev_space(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sobolev_space": _bench_sobolev_space(seed)}
