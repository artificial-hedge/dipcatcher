"""Craig interpolation on propositional implications (SYNTHETIC)."""

from __future__ import annotations

from quant_fund.models.sequent_calculus import eval_prop, valid_sequent, vars_of


def interpolant(a: tuple, b: tuple) -> tuple | None:
    """Craig: if A -> B valid, an interpolant uses only shared vars.
    For our toy: if A and B share a variable v such that A -> v -> B,
    return v; else T/true or F/false constants."""
    av, bv = vars_of(a), vars_of(b)
    shared = av & bv
    for v in sorted(shared):
        var = ("var", v)
        if valid_sequent([a], [var]) and valid_sequent([var], [b]):
            return var
    # check constant interpolants
    if valid_sequent([a], [("const", False)]):
        return ("const", False)
    if valid_sequent([("const", True)], [b]):
        return ("const", True)
    return None


def _eval_interp(f: tuple, env: dict[str, bool]) -> bool:
    if f[0] == "const":
        return bool(f[1])
    return eval_prop(f, env)


def _bench_interp_proof(seed: int = 0) -> float:
    checks = []
    a = ("var", "A")
    b = ("var", "B")
    c = ("var", "C")
    # (A & B) -> (A | C): interpolant A
    lhs = ("and", a, b)
    rhs = ("or", a, c)
    it = interpolant(lhs, rhs)
    checks.append(it == a)
    # (A & ~A) -> B: interpolant is bottom (lhs unsat)
    lhs2 = ("and", a, ("not", a))
    it2 = interpolant(lhs2, b)
    checks.append(it2 == ("const", False))
    # A -> (B -> A): interpolant is top (rhs tautology given A? share A)
    rhs3 = ("imp", b, a)
    it3 = interpolant(a, rhs3)
    checks.append(it3 == a or it3 == ("const", True))
    # no shared vars and neither side trivial -> A -> B has no interpolant
    checks.append(interpolant(a, b) is None)
    return float(sum(checks) / len(checks))


def bench_interp_proof(seed: int = 0) -> dict[str, float]:
    return {"synthetic_interp_proof": _bench_interp_proof(seed)}
