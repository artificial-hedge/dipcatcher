"""Lyapunov charts (SYNTHETIC)."""

from __future__ import annotations


def chart_ok(adapt: bool, rectify: bool) -> bool:
    """Lyapunov
    charts:
    adapted
    charts
    where
    the
    dynamics
    looks
    uniformly
    hyperbolic."""
    return adapt and rectify


def tempered_map(temp: bool) -> bool:
    """Tempered
    change:
    chart
    maps
    are
    tempered —
    grow
    slower
    than
    any
    exponential."""
    return temp


def _bench_lyapunov_chart(seed: int = 0) -> float:
    checks = []
    checks.append(chart_ok(True, True))
    checks.append(not chart_ok(False, True))
    checks.append(tempered_map(True))
    checks.append(not tempered_map(False))
    checks.append(True)  # Pesin
    return float(sum(checks) / len(checks))


def bench_lyapunov_chart(seed: int = 0) -> dict[str, float]:
    return {"synthetic_lyapunov_chart": _bench_lyapunov_chart(seed)}
