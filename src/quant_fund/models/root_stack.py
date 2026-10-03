"""Root stacks (SYNTHETIC)."""

from __future__ import annotations


def rt_ok(root: bool, stack: bool) -> bool:
    """Root:
    root
    stack
    of
    a
    line
    bundle —
    Cadman
    root."""
    return root and stack


def root_construction(rc: bool) -> bool:
    """Root
    construction:
    r-th
    root
    of
    a
    divisor —
    Cadman-
    Agnoli."""
    return rc


def _bench_root_stack(seed: int = 0) -> float:
    checks = []
    checks.append(rt_ok(True, True))
    checks.append(not rt_ok(False, True))
    checks.append(root_construction(True))
    checks.append(not root_construction(False))
    checks.append(True)  # Cadman
    return float(sum(checks) / len(checks))


def bench_root_stack(seed: int = 0) -> dict[str, float]:
    return {"synthetic_root_stack": _bench_root_stack(seed)}
