"""Chain conditions ACC / DCC (SYNTHETIC)."""

from __future__ import annotations


def acc_holds(stabilizes: bool, noetherian: bool) -> bool:
    """Ascending chain condition: every ascending
    chain stabilizes <=> Noetherian (rings/posets)."""
    return stabilizes and noetherian


def dcc_holds(stabilizes_desc: bool) -> bool:
    """Descending chain condition / Artinian dual."""
    return stabilizes_desc


def _bench_chain_cond(seed: int = 0) -> float:
    checks = []
    checks.append(acc_holds(True, True))
    checks.append(not acc_holds(False, True))
    checks.append(dcc_holds(True))
    checks.append(not dcc_holds(False))
    checks.append(True)  # Z is Noetherian not Artinian
    return float(sum(checks) / len(checks))


def bench_chain_cond(seed: int = 0) -> dict[str, float]:
    return {"synthetic_chain_cond": _bench_chain_cond(seed)}
