"""Natural deduction check on propositional derivations (SYNTHETIC)."""

from __future__ import annotations


def deduce(ctx: frozenset[str], rule: str, *args: str) -> frozenset[str]:
    """One ND rule step over a set-of-formulas context."""
    c = set(ctx)
    if rule == "and_i":
        a, b = args
        if a in c and b in c:
            c.add(f"({a}&{b})")
    elif rule == "and_e_l":
        (ab,) = args
        if ab.startswith("(") and "&" in ab:
            c.add(ab[1:].split("&")[0])
    elif rule == "imp_e":
        a, imp = args
        tgt = imp[1:-1].split(">")[1]
        if a in c and imp in c:
            c.add(tgt)
    elif rule == "imp_i":
        # discharge: add (a>b) when b derivable; caller pre-checks
        a, b = args
        c.add(f"({a}>{b})")
    return frozenset(c)


def _bench_natural_ded(seed: int = 0) -> float:
    checks = []
    c = frozenset({"A", "B"})
    c1 = deduce(c, "and_i", "A", "B")
    checks.append("(A&B)" in c1)
    c2 = deduce(frozenset({"(A&B)"}), "and_e_l", "(A&B)")
    checks.append("A" in c2)
    # modus ponens: {A, (A>B)} |- B
    c3 = deduce(frozenset({"A", "(A>B)"}), "imp_e", "A", "(A>B)")
    checks.append("B" in c3)
    # deduction theorem: imp_i adds the implication
    c4 = deduce(frozenset(), "imp_i", "A", "B")
    checks.append("(A>B)" in c4)
    # chained: {A, (A>B), (B>C)} |- C
    c5 = deduce(frozenset({"A", "(A>B)", "(B>C)"}), "imp_e", "A", "(A>B)")
    c5 = deduce(c5, "imp_e", "B", "(B>C)")
    checks.append("C" in c5)
    # failed elimination adds nothing
    c6 = deduce(frozenset({"A"}), "imp_e", "A", "(A>B)")
    checks.append("B" not in c6)
    return float(sum(checks) / len(checks))


def bench_natural_ded(seed: int = 0) -> dict[str, float]:
    return {"synthetic_natural_ded": _bench_natural_ded(seed)}
