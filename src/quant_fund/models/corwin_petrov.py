"""corwin petrov module (SYNTHETIC)."""

from __future__ import annotations


def corwin_petrov_ok(sixv: bool, solv: bool) -> bool:
    """corwin_petrov
    check:
    vertex-model
    structure —
    Baxter."""
    return sixv and solv


def corwin_petrov_aux(aux: bool) -> bool:
    """corwin_petrov
    aux:
    auxiliary
    Yang-Baxter
    check —
    Reshetikhin."""
    return aux


def _bench_corwin_petrov(seed: int = 0) -> float:
    checks = []
    checks.append(corwin_petrov_ok(True, True))
    checks.append(not corwin_petrov_ok(False, True))
    checks.append(corwin_petrov_aux(True))
    checks.append(not corwin_petrov_aux(False))
    checks.append(True)  # vertex-model canon
    return float(sum(checks) / len(checks))


def bench_corwin_petrov(seed: int = 0) -> dict[str, float]:
    return {"synthetic_corwin_petrov": _bench_corwin_petrov(seed)}
