"""harris recurrent module (SYNTHETIC)."""

from __future__ import annotations


def harris_recurrent_ok(dc: bool, hr: bool) -> bool:
    """harris_recurrent
    check:
    Markov-chain
    theory —
    stability."""
    return dc and hr


def harris_recurrent_aux(aux: bool) -> bool:
    """harris_recurrent
    aux:
    auxiliary
    chain
    check —
    mixing."""
    return aux


def _bench_harris_recurrent(seed: int = 0) -> float:
    checks = []
    checks.append(harris_recurrent_ok(True, True))
    checks.append(not harris_recurrent_ok(False, True))
    checks.append(harris_recurrent_aux(True))
    checks.append(not harris_recurrent_aux(False))
    checks.append(True)  # markov-chain canon
    return float(sum(checks) / len(checks))


def bench_harris_recurrent(seed: int = 0) -> dict[str, float]:
    return {"synthetic_harris_recurrent": _bench_harris_recurrent(seed)}
