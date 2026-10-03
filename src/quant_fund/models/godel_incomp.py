"""Godel-numbering and diagonal lemma scaffolding (SYNTHETIC)."""

from __future__ import annotations


def gnum(s: str) -> int:
    """A toy Godel numbering: encode each char to two digits."""
    return int("".join(f"{ord(c) % 100:02d}" for c in s) or "0")


def diag_step(f_expr: str) -> str:
    """One diagonalization step: substitute the formula's own code for
    its free variable marker '#'."""
    return f_expr.replace("#", str(gnum(f_expr)))


def fixed_point_check(f_expr: str) -> bool:
    """The diagonal lemma: F(diag(F)) where diag embeds gnum; a fixed
    point exists when applying diag_step yields a closed formula (no '#')."""
    return "#" not in diag_step(f_expr)


def _bench_godel_incomp(seed: int = 0) -> float:
    checks = []
    # encoding is injective on our alphabet
    checks.append(gnum("AB") != gnum("BA"))
    checks.append(gnum("x=#") == gnum("x=#"))
    # diagonalization substitutes the code
    d = diag_step("P(#)")
    checks.append("#" not in d)
    checks.append(str(gnum("P(#)")) in d)
    # fixed point exists for the toy provability predicate
    checks.append(fixed_point_check("Prov(#) -> ⊥"))
    # different formulas get different codes
    checks.append(gnum("P(#)") != gnum("Q(#)"))
    return float(sum(checks) / len(checks))


def bench_godel_incomp(seed: int = 0) -> dict[str, float]:
    return {"synthetic_godel_incomp": _bench_godel_incomp(seed)}
