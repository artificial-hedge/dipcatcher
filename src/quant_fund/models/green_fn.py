"""Green functions (SYNTHETIC)."""

from __future__ import annotations


def green_ok(pole: bool, vanish: bool) -> bool:
    """Green
    function:
    harmonic
    with
    logarithmic/
    Newtonian
    pole and
    zero
    boundary."""
    return pole and vanish


def symmetry(sym: bool) -> bool:
    """Green's
    identity:
    G(x,y)
    symmetric
    in
    the
    two
    arguments."""
    return sym


def _bench_green_fn(seed: int = 0) -> float:
    checks = []
    checks.append(green_ok(True, True))
    checks.append(not green_ok(False, True))
    checks.append(symmetry(True))
    checks.append(not symmetry(False))
    checks.append(True)  # Green
    return float(sum(checks) / len(checks))


def bench_green_fn(seed: int = 0) -> dict[str, float]:
    return {"synthetic_green_fn": _bench_green_fn(seed)}
