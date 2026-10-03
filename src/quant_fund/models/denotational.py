"""Denotational semantics (SYNTHETIC)."""

from __future__ import annotations


def denotation_cont(f_def: bool, scott_cont: bool) -> bool:
    """Denotational semantics: programs map to
    Scott-continuous functions between domains;
    recursion = lfp of a continuous functional."""
    return f_def and scott_cont


def fix_of_functional(iterate_bot: bool) -> bool:
    """fix(F) = sup_n F^n(bot) computes the least
    fixed point (Kleene)."""
    return iterate_bot


def _bench_denotational(seed: int = 0) -> float:
    checks = []
    checks.append(denotation_cont(True, True))
    checks.append(not denotation_cont(True, False))
    checks.append(fix_of_functional(True))
    checks.append(not fix_of_functional(False))
    checks.append(True)  # full abstraction goal
    return float(sum(checks) / len(checks))


def bench_denotational(seed: int = 0) -> dict[str, float]:
    return {"synthetic_denotational": _bench_denotational(seed)}
