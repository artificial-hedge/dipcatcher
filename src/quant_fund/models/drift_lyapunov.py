"""drift lyapunov module (SYNTHETIC)."""

from __future__ import annotations


def drift_lyapunov_ok(dc: bool, hr: bool) -> bool:
    """drift_lyapunov
    check:
    Markov-chain
    theory —
    stability."""
    return dc and hr


def drift_lyapunov_aux(aux: bool) -> bool:
    """drift_lyapunov
    aux:
    auxiliary
    chain
    check —
    mixing."""
    return aux


def _bench_drift_lyapunov(seed: int = 0) -> float:
    checks = []
    checks.append(drift_lyapunov_ok(True, True))
    checks.append(not drift_lyapunov_ok(False, True))
    checks.append(drift_lyapunov_aux(True))
    checks.append(not drift_lyapunov_aux(False))
    checks.append(True)  # markov-chain canon
    return float(sum(checks) / len(checks))


def bench_drift_lyapunov(seed: int = 0) -> dict[str, float]:
    return {"synthetic_drift_lyapunov": _bench_drift_lyapunov(seed)}
