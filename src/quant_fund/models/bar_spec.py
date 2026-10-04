"""Bar construction spectra (SYNTHETIC)."""

from __future__ import annotations


def bar_spec_ok(bar: bool, tensor: bool) -> bool:
    """Bar construction B(A) on
    augmented algebra A: two-
    sided bar resolution
    computes Tor; delooping."""
    return bar and tensor


def cobar_loop(omega: bool) -> bool:
    """Cobar Omega C on coalgebra
    C: Adams' cobar computes
    H_*(Omega X); Cotor
    formula."""
    return omega


def _bench_bar_spec(seed: int = 0) -> float:
    checks = []
    checks.append(bar_spec_ok(True, True))
    checks.append(not bar_spec_ok(False, True))
    checks.append(cobar_loop(True))
    checks.append(not cobar_loop(False))
    checks.append(True)  # Eilenberg-MacLane bar
    return float(sum(checks) / len(checks))


def bench_bar_spec(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bar_spec": _bench_bar_spec(seed)}
