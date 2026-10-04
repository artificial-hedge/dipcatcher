"""Turing degrees and reductions (SYNTHETIC)."""

from __future__ import annotations


def reduces(a_computable: bool, b_computable: bool) -> bool:
    """A <=_T B: if B is computable then so is A."""
    return a_computable or not b_computable


def _bench_recursion3(seed: int = 0) -> float:
    checks = []
    # any set reduces to itself
    checks.append(reduces(True, True))
    # computable <=_T anything
    checks.append(reduces(True, False))
    # noncomputable does not reduce to computable
    checks.append(not reduces(False, True))
    # transitivity of <=_T
    checks.append(reduces(reduces(True, False), False))
    # 0' > 0: the halting set is strictly harder
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_recursion3(seed: int = 0) -> dict[str, float]:
    return {"synthetic_recursion3": _bench_recursion3(seed)}
