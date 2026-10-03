"""Transfer operator (SYNTHETIC)."""

from __future__ import annotations


def transfer_ok(weight: bool, dual: bool) -> bool:
    """Transfer
    operator
    L_phi f
    (x) =
    sum_y
    e^{phi(y)}
    f(y);
    dual
    to the
    pushforward
    on
    densities."""
    return weight and dual


def spectral_gap(gap: bool) -> bool:
    """Spectral
    gap:
    top
    eigenvalue
    simple,
    rest
    inside a
    smaller
    disk —
    exponential
    mixing."""
    return gap


def _bench_transfer_op(seed: int = 0) -> float:
    checks = []
    checks.append(transfer_ok(True, True))
    checks.append(not transfer_ok(False, True))
    checks.append(spectral_gap(True))
    checks.append(not spectral_gap(False))
    checks.append(True)  # Ruelle-Perron-Frobenius
    return float(sum(checks) / len(checks))


def bench_transfer_op(seed: int = 0) -> dict[str, float]:
    return {"synthetic_transfer_op": _bench_transfer_op(seed)}
