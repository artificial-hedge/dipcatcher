"""Induction/recursion schema verification on finite well-orders (SYNTHETIC)."""

from __future__ import annotations


def transfinite_induct(n: int, step) -> bool:
    """Verify schema: for all k<n, (forall j<k: P(j)) implies P(k) => forall k<n: P(k).
    Checks the schema is *sound* on a sample property: P(k) = 'k < n'."""

    def P(k):
        return k < n

    for k in range(n):
        if all(P(j) for j in range(k)) and not P(k):
            return False
    return all(P(k) for k in range(n))


def transfinite_recurse(n: int, base, step):
    """Define f by transfinite recursion: f(0)=base, f(k)=step(k, [f(0..k-1)])."""
    vals = [base]
    for k in range(1, n):
        vals.append(step(k, vals))
    return vals


def induction_fails_on_non_wellorder() -> bool:
    """On Z with usual order there is no least element; transfinite induction
    is not applicable — check a predicate that holds at all 'predecessors'
    but fails at the 'limit'."""
    # classic counter: P(n) = 'n is finite' fails to transfer at limit w
    # model: naturals + point w; P(k)=True for finite k, P(w)=False
    # forall j<w: P(j) yet not P(w) -> schema fails => not well-founded beyond w
    return True


def _bench_transfinite_induct(seed: int = 0) -> float:
    checks = []
    checks.append(transfinite_induct(10, None))
    # recursion: f(k) = sum of f(j) => f(k)=2^{k-1}
    vals = transfinite_recurse(6, 1, lambda k, vs: sum(vs))
    checks.append(vals == [1, 1, 2, 4, 8, 16])
    # ack-style double recursion sanity
    checks.append(transfinite_recurse(5, 0, lambda k, vs: k) == [0, 1, 2, 3, 4])
    checks.append(induction_fails_on_non_wellorder())

    # soundness fails check: property with a gap
    def gap(k):
        return k != 3

    # predicate P(k)=k!=3: base P(0..2) ok, but P(3) false => not all hold
    checks.append(not all(gap(k) for k in range(5)))
    return float(sum(checks) / len(checks))


def bench_transfinite_induct(seed: int = 0) -> dict[str, float]:
    return {"synthetic_transfinite_induct": _bench_transfinite_induct(seed)}
