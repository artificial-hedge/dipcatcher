"""Double limits (SYNTHETIC)."""

from __future__ import annotations


def dl_ok(double: bool, limit: bool) -> bool:
    """Double
    limit:
    double
    limit —
    weighted
    double."""
    return double and limit


def tabulator(tb: bool) -> bool:
    """Tabulator:
    tabulator —
    double
    tabulator."""
    return tb


def _bench_double_lim(seed: int = 0) -> float:
    checks = []
    checks.append(dl_ok(True, True))
    checks.append(not dl_ok(False, True))
    checks.append(tabulator(True))
    checks.append(not tabulator(False))
    checks.append(True)  # Grandis-Pare
    return float(sum(checks) / len(checks))


def bench_double_lim(seed: int = 0) -> dict[str, float]:
    return {"synthetic_double_lim": _bench_double_lim(seed)}
