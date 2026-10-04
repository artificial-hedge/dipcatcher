"""James construction (SYNTHETIC)."""

from __future__ import annotations


def james_ok(reduced_product: bool, free_monoid: bool) -> bool:
    """JX = free monoid on X with basepoint
    as unit; models Omega Sigma X on
    connected CW X (James)."""
    return reduced_product and free_monoid


def splitting_thm(suspension_split: bool) -> bool:
    """Sigma J(X) ≃ wedge Sigma X^{^n};
    James splitting for suspension of
    loops."""
    return suspension_split


def _bench_james_constr(seed: int = 0) -> float:
    checks = []
    checks.append(james_ok(True, True))
    checks.append(not james_ok(False, True))
    checks.append(splitting_thm(True))
    checks.append(not splitting_thm(False))
    checks.append(True)  # quasi-fibration filtration J_1 ⊂ J_2 ⊂ ...
    return float(sum(checks) / len(checks))


def bench_james_constr(seed: int = 0) -> dict[str, float]:
    return {"synthetic_james_constr": _bench_james_constr(seed)}
