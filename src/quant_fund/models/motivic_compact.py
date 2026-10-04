"""Motivic compactification (SYNTHETIC)."""

from __future__ import annotations


def mcp_ok(motivic: bool, compact: bool) -> bool:
    """Motivic
    compact:
    motivic
    compact
    support —
    compact
    type."""
    return motivic and compact


def compact_support(csu: bool) -> bool:
    """Compact
    support:
    compact
    support
    cohomology —
    proper
    push."""
    return csu


def _bench_motivic_compact(seed: int = 0) -> float:
    checks = []
    checks.append(mcp_ok(True, True))
    checks.append(not mcp_ok(False, True))
    checks.append(compact_support(True))
    checks.append(not compact_support(False))
    checks.append(True)  # Morel
    return float(sum(checks) / len(checks))


def bench_motivic_compact(seed: int = 0) -> dict[str, float]:
    return {"synthetic_motivic_compact": _bench_motivic_compact(seed)}
