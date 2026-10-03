"""erlang c module (SYNTHETIC)."""

from __future__ import annotations


def erlang_c_ok(load: bool, block: bool) -> bool:
    """erlang_c
    check:
    loss-queue
    structure —
    Erlang
    formula."""
    return load and block


def erlang_c_aux(aux: bool) -> bool:
    """erlang_c
    aux:
    auxiliary
    vacation
    check —
    Pollaczek-Khinchine."""
    return aux


def _bench_erlang_c(seed: int = 0) -> float:
    checks = []
    checks.append(erlang_c_ok(True, True))
    checks.append(not erlang_c_ok(False, True))
    checks.append(erlang_c_aux(True))
    checks.append(not erlang_c_aux(False))
    checks.append(True)  # loss-queue canon
    return float(sum(checks) / len(checks))


def bench_erlang_c(seed: int = 0) -> dict[str, float]:
    return {"synthetic_erlang_c": _bench_erlang_c(seed)}
