"""Cut-free LK proof search on propositional sequents (SYNTHETIC)."""

from __future__ import annotations


def eval_prop(f: tuple, env: dict[str, bool]) -> bool:
    """Evaluate a formula tree: ('var',x), ('not',f), ('and'|'or'|'imp',a,b)."""
    kind = f[0]
    if kind == "const":
        return bool(f[1])
    if kind == "var":
        return env[f[1]]
    if kind == "not":
        return not eval_prop(f[1], env)
    a, b = f[1], f[2]
    va, vb = eval_prop(a, env), eval_prop(b, env)
    if kind == "and":
        return va and vb
    if kind == "or":
        return va or vb
    return (not va) or vb


def vars_of(f: tuple) -> set[str]:
    if f[0] == "const":
        return set()
    if f[0] == "var":
        return {f[1]}
    if f[0] == "not":
        return vars_of(f[1])
    return vars_of(f[1]) | vars_of(f[2])


def valid_sequent(gamma: list[tuple], delta: list[tuple]) -> bool:
    """Gamma |- Delta valid iff every valuation making all of Gamma
    true makes some Delta true (invertible rules of LK decide this)."""
    vs: set[str] = set()
    for f in gamma + delta:
        vs |= vars_of(f)
    vs_list = sorted(vs)
    for mask in range(1 << len(vs_list)):
        env = {v: bool(mask & (1 << i)) for i, v in enumerate(vs_list)}
        if all(eval_prop(g, env) for g in gamma) and not any(eval_prop(d, env) for d in delta):
            return False
    return True


def _bench_sequent_calculus(seed: int = 0) -> float:
    checks = []
    a = ("var", "A")
    b = ("var", "B")
    # |- A v ~A
    checks.append(valid_sequent([], [("or", a, ("not", a))]))
    # |- (A /\ B) -> A
    checks.append(valid_sequent([], [("imp", ("and", a, b), a)]))
    # A -> B, A |- B (modus ponens shape)
    checks.append(valid_sequent([("imp", a, b), a], [b]))
    # cut-free completeness on a nontrivial identity
    checks.append(
        valid_sequent(
            [("or", a, b)],
            [("not", ("and", ("not", a), ("not", b)))],
        )
    )
    # invalid sequent A |- B
    checks.append(not valid_sequent([a], [b]))
    # weakening-valid: A, B |- A
    checks.append(valid_sequent([a, b], [a]))
    return float(sum(checks) / len(checks))


def bench_sequent_calculus(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sequent_calculus": _bench_sequent_calculus(seed)}
